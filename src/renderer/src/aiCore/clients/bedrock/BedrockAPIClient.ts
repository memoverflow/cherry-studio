import {
  BedrockRuntimeClient,
  ContentBlock,
  ContentBlockDelta,
  ConversationRole,
  ConverseCommand,
  ConverseCommandOutput,
  ConverseResponse,
  ConverseStreamCommand,
  InferenceConfiguration,
  TokenUsage,
  ToolConfiguration
} from '@aws-sdk/client-bedrock-runtime'
import { DEFAULT_MAX_TOKENS } from '@renderer/config/constant'
import { findTokenLimit, isReasoningModel, isVisionModel } from '@renderer/config/models'
import { getAssistantSettings } from '@renderer/services/AssistantService'
import { estimateTextTokens } from '@renderer/services/TokenService'
import {
  Assistant,
  EFFORT_RATIO,
  FileTypes,
  MCPCallToolResponse,
  MCPTool,
  MCPToolResponse,
  Model,
  Provider,
  ToolCallResponse
} from '@renderer/types'
import {
  ChunkType,
  TextCompleteChunk,
  TextStartChunk,
  ThinkingDeltaChunk,
  ThinkingStartChunk
} from '@renderer/types/chunk'
import { Message } from '@renderer/types/newMessage'
import {
  BedrockSdkInstance,
  BedrockSdkMessageParam,
  BedrockSdkParams,
  BedrockSdkRawChunk,
  BedrockSdkRawOutput,
  BedrockSdkTool,
  BedrockSdkToolCall,
  SdkModel
} from '@renderer/types/sdk'
import { addImageFileToContents } from '@renderer/utils/formats'
import { findFileBlocks, findImageBlocks } from '@renderer/utils/messageUtils/find'
import { buildSystemPrompt } from '@renderer/utils/prompt'
import { Buffer } from 'buffer'

import { GenericChunk } from '../../middleware/schemas'
import { BaseApiClient } from '../BaseApiClient'
import { RequestTransformer, ResponseChunkTransformer } from '../types'

/**
 * 工具调用状态枚举
 */
enum ToolCallState {
  /** 等待开始 */
  PENDING = 'pending',
  /** 正在累积输入 */
  ACCUMULATING = 'accumulating',
  /** 输入完整但未发送 */
  COMPLETE = 'complete',
  /** 已发送 */
  SENT = 'sent',
  /** 调用失败 */
  FAILED = 'failed',
  /** 已取消（重试超限） */
  CANCELLED = 'cancelled'
}

/**
 * 扩展的工具调用接口，包含状态管理
 */
interface ManagedToolCall extends BedrockSdkToolCall {
  /** 工具调用状态 */
  state: ToolCallState
  /** 累积的输入字符串 */
  inputBuffer: string
  /** 失败重试次数 */
  retryCount: number
  /** 最后一次错误信息 */
  lastError?: string
}

/**
 * 流式响应状态管理器
 * 用于跟踪流式响应的处理状态，包括文本输出、思考内容和工具调用
 */
class StreamState {
  /** 是否已开始文本输出 */
  hasStartedText = false
  /** 是否已开始思考输出 */
  hasStartedThinking = false
  /** 思考开始时间 */
  thinkingStartTime?: number
  /** 响应停止原因 */
  stopReason?: string
  /** 使用情况统计 */
  usage?: TokenUsage
  /** 管理的工具调用列表 */
  managedToolCalls: ManagedToolCall[] = []
  /** 累积的思考内容 */
  thinkingContent = ''
  /** 思考签名 */
  thinkingSignature = ''
  /** 是否有工具调用已发送 */
  hasToolCallsSent = false
  /** 累积的token使用量 */
  accumulatedUsage = { inputTokens: 0, outputTokens: 0, totalTokens: 0 }
  /** 工具调用失败重试次数 */
  toolCallRetryCount = 0
  /** 最大重试次数 */
  maxRetryCount = 3

  /**
   * 重置状态到初始值
   */
  reset(): void {
    this.hasStartedText = false
    this.hasStartedThinking = false
    this.thinkingStartTime = undefined
    this.stopReason = undefined
    this.usage = undefined
    this.managedToolCalls = []
    this.thinkingContent = ''
    this.thinkingSignature = ''
    this.hasToolCallsSent = false
    this.toolCallRetryCount = 0
  }

  /**
   * 检查是否应该停止工具调用重试
   * @returns 是否应该停止重试
   */
  shouldStopRetrying(): boolean {
    return this.toolCallRetryCount >= this.maxRetryCount
  }

  /**
   * 增加重试次数
   */
  incrementRetryCount(): void {
    this.toolCallRetryCount++
  }

  /**
   * 累加token使用量
   * @param usage - 使用量统计
   */
  addUsage(usage: TokenUsage): void {
    this.accumulatedUsage.inputTokens += usage.inputTokens || 0
    this.accumulatedUsage.outputTokens += usage.outputTokens || 0
    this.accumulatedUsage.totalTokens += usage.totalTokens || 0
  }

  /**
   * 获取最后一个工具调用
   * @returns 最后一个工具调用或undefined
   */
  getLastToolCall(): ManagedToolCall | undefined {
    return this.managedToolCalls[this.managedToolCalls.length - 1]
  }

  /**
   * 添加新的工具调用
   * @param toolCall - 工具调用信息
   */
  addToolCall(toolCall: Omit<BedrockSdkToolCall, 'input'>): void {
    const managedToolCall: ManagedToolCall = {
      ...toolCall,
      input: {},
      state: ToolCallState.PENDING,
      inputBuffer: '',
      retryCount: 0
    }
    this.managedToolCalls.push(managedToolCall)
  }

  /**
   * 标记工具调用为失败状态
   * @param toolCall - 工具调用
   * @param error - 错误信息
   */
  markToolCallAsFailed(toolCall: ManagedToolCall, error: string): void {
    toolCall.retryCount++
    toolCall.lastError = error
    toolCall.state = toolCall.retryCount >= this.maxRetryCount ? ToolCallState.CANCELLED : ToolCallState.FAILED
  }

  /**
   * 获取所有失败的工具调用
   * @returns 失败的工具调用列表
   */
  getFailedToolCalls(): ManagedToolCall[] {
    return this.managedToolCalls.filter((tool) => tool.state === ToolCallState.FAILED)
  }

  /**
   * 获取所有被取消的工具调用（重试超限）
   * @returns 被取消的工具调用列表
   */
  getCancelledToolCalls(): ManagedToolCall[] {
    return this.managedToolCalls.filter((tool) => tool.state === ToolCallState.CANCELLED)
  }

  /**
   * 获取所有已完成的工具调用
   * @returns 已完成的工具调用列表
   */
  getCompletedToolCalls(): BedrockSdkToolCall[] {
    return this.managedToolCalls
      .filter((tool) => tool.state === ToolCallState.COMPLETE || tool.state === ToolCallState.SENT)
      .map((tool) => ({
        toolUseId: tool.toolUseId,
        name: tool.name,
        input: tool.input,
        type: tool.type
      }))
  }
}

/**
 * Bedrock流式响应处理器
 * 负责处理所有类型的流式响应事件
 */
class StreamProcessor {
  constructor(private client: BedrockAPIClient) {}

  /**
   * 处理单个流式响应事件
   * @param event - 流式响应事件
   * @param state - 流状态管理器
   * @param controller - 流控制器
   * @returns 是否成功处理事件
   */
  processEvent(
    event: BedrockSdkRawChunk,
    state: StreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): boolean {
    if (!event) return false

    // 处理非流式响应
    if ('output' in event && 'stopReason' in event) {
      return this.handleNonStreamResponse(event as ConverseResponse, state, controller)
    }

    // 处理流式事件
    if ('messageStart' in event) {
      state.reset()
      return true
    }

    if ('contentBlockStart' in event) {
      return this.handleContentBlockStart(event, state)
    }

    if ('contentBlockDelta' in event) {
      return this.handleContentBlockDelta(event, state, controller)
    }

    if ('contentBlockStop' in event) {
      return this.handleContentBlockStop(event, state, controller)
    }

    if ('messageStop' in event) {
      state.stopReason = (event as any).stopReason
      return true
    }

    if ('metadata' in event && (event as any).metadata?.usage) {
      return this.handleMetadata(event, state, controller)
    }

    return false
  }

  /**
   * 处理内容块开始事件
   * @param event - 事件数据
   * @param state - 流状态
   * @returns 是否成功处理
   */
  private handleContentBlockStart(event: BedrockSdkRawChunk, state: StreamState): boolean {
    const startEvent = (event as any).contentBlockStart
    if (!startEvent?.start) return false

    return ContentBlock.visit(startEvent.start, {
      text: () => false,
      image: () => false,
      document: () => false,
      video: () => false,
      toolUse: (toolUseStart) => {
        state.addToolCall({
          toolUseId: toolUseStart.toolUseId || `tool_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`,
          name: toolUseStart.name || 'unknown_tool',
          type: 'tool_use'
        })
        return true
      },
      toolResult: () => false,
      guardContent: () => false,
      cachePoint: () => false,
      reasoningContent: () => false,
      citationsContent: () => false,
      _: () => false
    })
  }

  /**
   * 处理内容块增量事件
   * @param event - 事件数据
   * @param state - 流状态
   * @param controller - 流控制器
   * @returns 是否成功处理
   */
  private handleContentBlockDelta(
    event: BedrockSdkRawChunk,
    state: StreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): boolean {
    const deltaEvent = (event as any).contentBlockDelta
    if (!deltaEvent?.delta) return false

    return ContentBlockDelta.visit(deltaEvent.delta, {
      text: (text: string) => {
        this.handleTextDelta(text, state, controller)
        return true
      },
      reasoningContent: (reasoningDelta) => {
        if ('text' in reasoningDelta && reasoningDelta.text) {
          this.handleThinkingDelta(reasoningDelta.text, state, controller)
        }
        if ('signature' in reasoningDelta && reasoningDelta.signature) {
          state.thinkingSignature += reasoningDelta.signature
        }
        return true
      },
      toolUse: (toolUseDelta) => {
        console.log('[BedrockApiClient] toolUseDelta', toolUseDelta)
        console.log('[BedrockApiClient] toolUseDelta.input', toolUseDelta.input)
        this.handleToolUseDelta(toolUseDelta.input || '', state, controller)
        return true
      },
      citation: () => false,
      _: () => false
    })
  }

  /**
   * 处理内容块停止事件
   * @param event - 事件数据
   * @param state - 流状态
   * @param controller - 流控制器
   * @returns 是否成功处理
   */
  private handleContentBlockStop(
    event: BedrockSdkRawChunk,
    state: StreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): boolean {
    const stopEvent = (event as any).contentBlockStop
    if (!stopEvent) return false

    // 处理思考块结束
    if (stopEvent.contentBlockIndex === 0 && state.hasStartedThinking) {
      controller.enqueue({ type: ChunkType.TEXT_COMPLETE, text: '' } as TextCompleteChunk)
    }

    // 处理工具调用块结束，完成所有待完成的工具调用
    this.finalizeAllPendingToolCalls(state, controller)

    return true
  }

  /**
   * 处理元数据事件
   * @param event - 事件数据
   * @param state - 流状态
   * @param controller - 流控制器
   * @returns 是否成功处理
   */
  private handleMetadata(
    event: BedrockSdkRawChunk,
    state: StreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): boolean {
    const metadataEvent = (event as any).metadata
    state.usage = metadataEvent.usage

    if (state.usage) {
      this.client.addToGlobalUsage(state.usage)
      state.addUsage(state.usage)
    }

    // 保存思考块
    if (state.thinkingContent.trim() && state.thinkingSignature.trim()) {
      this.client.saveThinkingBlock({
        thinking: state.thinkingContent.trim(),
        signature: state.thinkingSignature.trim()
      })
    }

    this.finalizeResponse(state, controller)
    return true
  }

  /**
   * 处理非流式响应
   * @param response - 响应数据
   * @param state - 流状态
   * @param controller - 流控制器
   * @returns 是否成功处理
   */
  private handleNonStreamResponse(
    response: ConverseResponse,
    state: StreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): boolean {
    if (response.usage) state.usage = response.usage

    const messageContent = response.output?.message?.content
    if (!messageContent || !Array.isArray(messageContent)) return false

    let hasContent = false
    messageContent.forEach((contentBlock) => {
      ContentBlock.visit(contentBlock, {
        text: (text: string) => {
          this.handleTextDelta(text, state, controller)
          hasContent = true
        },
        image: () => {},
        document: () => {},
        video: () => {},
        toolUse: (toolUseBlock) => {
          state.addToolCall({
            toolUseId: toolUseBlock.toolUseId || `tool_${Date.now()}`,
            name: toolUseBlock.name || 'unknown_tool',
            type: 'tool_use'
          })
          const lastTool = state.getLastToolCall()
          if (lastTool) {
            lastTool.input = toolUseBlock.input || {}
            lastTool.state = ToolCallState.COMPLETE
          }
          hasContent = true
        },
        toolResult: () => {},
        guardContent: () => {},
        cachePoint: () => {},
        reasoningContent: () => {},
        citationsContent: () => {},
        _: () => {}
      })
    })

    if (hasContent) {
      state.stopReason = response.stopReason
      this.finalizeResponse(state, controller)
    }

    return hasContent
  }

  /**
   * 处理文本增量
   * @param text - 文本内容
   * @param state - 流状态
   * @param controller - 流控制器
   */
  private handleTextDelta(
    text: string,
    state: StreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): void {
    if (!state.hasStartedText) {
      controller.enqueue({ type: ChunkType.TEXT_START } as TextStartChunk)
      state.hasStartedText = true
    }
    controller.enqueue({ type: ChunkType.TEXT_DELTA, text })
  }

  /**
   * 处理思考增量
   * @param text - 思考文本
   * @param state - 流状态
   * @param controller - 流控制器
   */
  private handleThinkingDelta(
    text: string,
    state: StreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): void {
    if (!state.hasStartedThinking) {
      controller.enqueue({ type: ChunkType.THINKING_START } as ThinkingStartChunk)
      state.hasStartedThinking = true
      state.thinkingContent = ''
      state.thinkingStartTime = Date.now()
    }

    state.thinkingContent += text
    controller.enqueue({ type: ChunkType.THINKING_DELTA, text } as ThinkingDeltaChunk)
  }

  /**
   * 处理工具使用增量
   * @param inputDelta - 输入增量
   * @param state - 流状态
   * @param controller - 流控制器
   */
  private handleToolUseDelta(
    inputDelta: string,
    state: StreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): void {
    const lastTool = state.getLastToolCall()
    if (!lastTool) return

    // 更新工具状态
    if (lastTool.state === ToolCallState.PENDING) {
      lastTool.state = ToolCallState.ACCUMULATING
    }

    // 累积输入
    lastTool.inputBuffer += inputDelta

    // 尝试解析并发送完整的工具调用
    this.tryCompleteToolCall(lastTool, controller, state)
  }

  /**
   * 尝试完成工具调用
   * @param toolCall - 工具调用
   * @param controller - 流控制器
   * @param state - 流状态管理器
   */
  private tryCompleteToolCall(
    toolCall: ManagedToolCall,
    controller: TransformStreamDefaultController<GenericChunk>,
    state: StreamState
  ): void {
    if (toolCall.state !== ToolCallState.ACCUMULATING) return

    const [isComplete, parsedInput] = this.isToolInputComplete(toolCall.inputBuffer)

    if (isComplete) {
      toolCall.input = parsedInput
      toolCall.state = ToolCallState.COMPLETE

      try {
        // 立即发送工具调用
        controller.enqueue({
          type: ChunkType.MCP_TOOL_CREATED,
          tool_calls: [
            {
              toolUseId: toolCall.toolUseId,
              name: toolCall.name,
              input: toolCall.input,
              type: toolCall.type
            }
          ]
        })

        toolCall.state = ToolCallState.SENT
      } catch (error) {
        const errorMessage = `发送工具调用失败: ${error instanceof Error ? error.message : String(error)}`
        console.error('[BedrockAPI]', errorMessage)

        // 使用 StreamState 的统一方法标记失败
        state.markToolCallAsFailed(toolCall, errorMessage)
      }
    } else {
      // 检查输入缓冲区是否过长或包含明显错误的 JSON
      if (toolCall.inputBuffer.length > 10000) {
        const errorMessage = '工具调用输入过长，可能存在解析问题'
        console.warn('[BedrockAPI]', errorMessage)

        // 使用 StreamState 的统一方法标记失败
        state.markToolCallAsFailed(toolCall, errorMessage)
      }
    }
  }

  /**
   * 完成所有待完成的工具调用
   * @param state - 流状态
   * @param controller - 流控制器
   */
  private finalizeAllPendingToolCalls(
    state: StreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): void {
    const unsentTools: BedrockSdkToolCall[] = []

    for (const tool of state.managedToolCalls) {
      if (tool.state === ToolCallState.ACCUMULATING) {
        // 强制完成工具调用
        const [, parsedInput] = this.isToolInputComplete(tool.inputBuffer, true)
        tool.input = parsedInput
        tool.state = ToolCallState.COMPLETE
      }

      if (tool.state === ToolCallState.COMPLETE) {
        unsentTools.push({
          toolUseId: tool.toolUseId,
          name: tool.name,
          input: tool.input,
          type: tool.type
        })
        tool.state = ToolCallState.SENT
      }
    }

    // 批量发送所有未发送的工具调用
    if (unsentTools.length > 0) {
      controller.enqueue({
        type: ChunkType.MCP_TOOL_CREATED,
        tool_calls: unsentTools
      })
    }
  }

  /**
   * 检查工具输入是否完整
   * @param input - 输入字符串
   * @param forceComplete - 是否强制完成
   * @returns [是否完整, 解析后的输入]
   */
  private isToolInputComplete(input: string, forceComplete = false): [boolean, any] {
    // 空字符串表示无参数工具调用
    if (input === '') return [true, {}]

    if (input.trim()) {
      try {
        return [true, JSON.parse(input)]
      } catch {
        // 如果强制完成，返回空对象；否则继续等待
        return forceComplete ? [true, {}] : [false, {}]
      }
    }

    return forceComplete ? [true, {}] : [false, {}]
  }

  /**
   * 完成响应处理
   * @param state - 流状态
   * @param controller - 流控制器
   */
  private finalizeResponse(state: StreamState, controller: TransformStreamDefaultController<GenericChunk>): void {
    // 先检查是否有工具调用需要处理，不依赖 stopReason
    if (state.managedToolCalls.length > 0) {
      // 检查是否有工具调用失败需要重试
      const failedTools = state.getFailedToolCalls()
      const cancelledTools = state.getCancelledToolCalls()

      if (cancelledTools.length > 0) {
        // 有工具调用达到最大重试次数，发送错误信息并结束
        const errorMessage = `工具调用失败，已达到最大重试次数(${state.maxRetryCount})次。失败的工具: ${cancelledTools.map((t) => `${t.name}(${t.lastError})`).join(', ')}`
        console.error('[BedrockAPI]', errorMessage)

        controller.enqueue({
          type: ChunkType.TEXT_START
        })
        controller.enqueue({
          type: ChunkType.TEXT_DELTA,
          text: `⚠️ ${errorMessage}`
        })

        // 发送使用统计并结束
        this.emitUsageStatistics(controller)
        return
      }
      if (failedTools.length > 0) {
        // 检查是否还有可以重试的工具调用（未达到单个工具的最大重试次数）
        const retryableTools = failedTools.filter((tool) => tool.retryCount < state.maxRetryCount)

        if (retryableTools.length > 0) {
          // 有失败的工具调用但未达到重试上限，准备重试
          state.incrementRetryCount()
          console.warn(
            `[BedrockAPI] 工具调用失败，准备第 ${state.toolCallRetryCount} 次重试，可重试工具数量: ${retryableTools.length}`
          )

          // 重置可重试的工具调用状态以便重试
          retryableTools.forEach((tool) => {
            tool.state = ToolCallState.PENDING
            tool.inputBuffer = ''
          })

          // 重新发送可重试的工具调用
          const retryToolCalls = retryableTools.map((tool) => ({
            toolUseId: tool.toolUseId,
            name: tool.name,
            input: tool.input,
            type: tool.type
          }))

          controller.enqueue({
            type: ChunkType.MCP_TOOL_CREATED,
            tool_calls: retryToolCalls
          })

          state.hasToolCallsSent = true
          return
        }
      }

      // 确保所有工具调用都已发送
      this.finalizeAllPendingToolCalls(state, controller)
      state.hasToolCallsSent = true
    }

    // 发送使用统计
    this.emitUsageStatistics(controller)
  }

  /**
   * 发送使用统计信息
   * @param controller - 流控制器
   */
  private emitUsageStatistics(controller: TransformStreamDefaultController<GenericChunk>): void {
    const globalUsage = this.client.getGlobalUsage()
    if (globalUsage.totalTokens > 0) {
      controller.enqueue({
        type: ChunkType.LLM_RESPONSE_COMPLETE,
        response: {
          usage: {
            prompt_tokens: globalUsage.inputTokens || 0,
            completion_tokens: globalUsage.outputTokens || 0,
            total_tokens: globalUsage.totalTokens || 0
          }
        }
      })
    }
  }
}

/**
 * Amazon Bedrock API 客户端
 *
 * 这是一个重构的Bedrock API客户端，特点：
 * - 充分利用AWS SDK的原生类和接口
 * - 支持流式和非流式响应处理
 * - 实现智能的工具调用管理
 * - 提供推理模型的思考模式支持
 * - 集成Cherry Studio的业务逻辑
 */
export class BedrockAPIClient extends BaseApiClient<
  BedrockSdkInstance,
  BedrockSdkParams,
  BedrockSdkRawOutput,
  BedrockSdkRawChunk,
  BedrockSdkMessageParam,
  BedrockSdkToolCall,
  BedrockSdkTool
> {
  /** 默认AWS区域 */
  private static readonly DEFAULT_REGION = 'us-east-1'
  /** 最小思考预算令牌数 */
  private static readonly MIN_THINKING_BUDGET_TOKENS = 1024

  /** Bedrock Runtime客户端实例 */
  private bedrockClient?: BedrockRuntimeClient
  /** 最近一次响应的思考块数据 */
  private lastThinkingBlock: { thinking: string; signature: string } | null = null
  /** 全局累积的token使用量 */
  private globalAccumulatedUsage = { inputTokens: 0, outputTokens: 0, totalTokens: 0 }

  constructor(provider: Provider) {
    super(provider)
  }

  /**
   * 生成图像功能（暂不支持）
   * @returns 空数组
   */
  override async generateImage(): Promise<string[]> {
    console.warn('[BedrockAPI] Bedrock SDK暂不支持图像生成功能')
    return []
  }

  /**
   * 获取嵌入向量维度（暂不支持）
   * @throws Error 抛出不支持错误
   */
  override async getEmbeddingDimensions(): Promise<number> {
    throw new Error('[BedrockAPI] Bedrock SDK暂不支持嵌入向量功能')
  }

  /**
   * 列出可用模型（暂未实现）
   * @returns 空数组
   */
  override async listModels(): Promise<SdkModel[]> {
    console.warn('[BedrockAPI] 模型列表功能尚未实现')
    return []
  }

  /**
   * 创建聊天完成请求
   * @param payload - 请求参数
   * @returns 响应输出
   */
  override async createCompletions(payload: BedrockSdkParams): Promise<BedrockSdkRawOutput> {
    const client = await this.getSdkInstance()
    const messages = this.transformMessagesForBedrock(payload.messages)

    const commandParams = {
      modelId: payload.modelId,
      messages,
      system: payload.system,
      inferenceConfig: payload.inferenceConfig,
      toolConfig: payload.toolConfig,
      additionalModelRequestFields: payload.additionalModelRequestFields
    }

    try {
      if (payload.stream) {
        const command = new ConverseStreamCommand(commandParams)
        const response = await client.send(command)
        return response.stream as BedrockSdkRawOutput
      } else {
        const command = new ConverseCommand(commandParams)
        const response = await client.send(command)
        return response as BedrockSdkRawOutput
      }
    } catch (error) {
      console.error('[BedrockAPI] 创建完成请求失败:', error)
      throw error
    }
  }

  /**
   * 获取Bedrock SDK实例（懒加载）
   * @returns SDK实例
   */
  async getSdkInstance(): Promise<BedrockSdkInstance> {
    if (!this.bedrockClient) {
      this.bedrockClient = this.createBedrockRuntimeClient()
    }
    return this.bedrockClient as BedrockSdkInstance
  }

  /**
   * 创建Bedrock Runtime客户端
   * @returns 客户端实例
   */
  private createBedrockRuntimeClient(): BedrockRuntimeClient {
    if (this.provider.type !== 'bedrock') {
      throw new Error('[BedrockAPI] 提供商类型必须是bedrock')
    }

    return new BedrockRuntimeClient({
      region: this.provider.region || BedrockAPIClient.DEFAULT_REGION,
      credentials: {
        accessKeyId: this.provider.accessKey || '',
        secretAccessKey: this.provider.secretKey || ''
      }
    })
  }

  /**
   * 获取模型ID（支持跨区域）
   * @param model - 模型信息
   * @returns 模型ID
   */
  private getModelId(model: Model): string {
    if (this.provider.type !== 'bedrock') return model.id
    return this.provider.crossRegion ? `us.${model.id}` : model.id
  }

  /**
   * 获取温度参数
   * @param assistant - 助手配置
   * @param model - 模型信息
   * @returns 温度值或undefined
   */
  override getTemperature(assistant: Assistant, model: Model): number | undefined {
    if (assistant.settings?.reasoning_effort && isReasoningModel(model)) {
      return undefined
    }
    return assistant.settings?.temperature
  }

  /**
   * 获取TopP参数
   * @param assistant - 助手配置
   * @param model - 模型信息
   * @returns TopP值或undefined
   */
  override getTopP(assistant: Assistant, model: Model): number | undefined {
    if (assistant.settings?.reasoning_effort && isReasoningModel(model)) {
      return undefined
    }
    return assistant.settings?.topP
  }

  /**
   * 保存思考块数据
   * @param thinkingBlock - 思考块数据
   */
  public saveThinkingBlock(thinkingBlock: { thinking: string; signature: string }): void {
    this.lastThinkingBlock = thinkingBlock
  }

  /**
   * 添加到全局累积使用量
   * @param usage - 使用量统计
   */
  public addToGlobalUsage(usage: TokenUsage): void {
    this.globalAccumulatedUsage.inputTokens += usage.inputTokens || 0
    this.globalAccumulatedUsage.outputTokens += usage.outputTokens || 0
    this.globalAccumulatedUsage.totalTokens += usage.totalTokens || 0
  }

  /**
   * 获取全局累积使用量
   * @returns 累积使用量
   */
  public getGlobalUsage(): { inputTokens: number; outputTokens: number; totalTokens: number } {
    return { ...this.globalAccumulatedUsage }
  }

  /**
   * 重置全局累积使用量
   */
  public resetGlobalUsage(): void {
    this.globalAccumulatedUsage = { inputTokens: 0, outputTokens: 0, totalTokens: 0 }
  }

  /**
   * 处理工具调用数据
   * @param toolCalls - 工具调用列表
   * @returns 处理后的工具调用列表
   */
  public processToolCalls(toolCalls: BedrockSdkToolCall[]): BedrockSdkToolCall[] {
    return toolCalls.map((toolCall) => {
      try {
        const parsedInput =
          typeof toolCall.input === 'string' && toolCall.input ? JSON.parse(toolCall.input) : toolCall.input
        return { ...toolCall, input: parsedInput || {} }
      } catch (error) {
        console.error('[BedrockAPI] 解析工具调用输入失败:', toolCall.input, error)
        return { ...toolCall, input: {} }
      }
    })
  }

  /**
   * 处理工具调用错误
   * @param error - 错误信息
   * @param toolCall - 失败的工具调用
   * @param state - 流状态管理器
   * @returns 是否应该重试
   */
  public handleToolCallError(error: string, toolCall: BedrockSdkToolCall, state: StreamState): boolean {
    // 查找对应的管理工具调用
    const managedTool = state.managedToolCalls.find((t) => t.toolUseId === toolCall.toolUseId)
    if (!managedTool) {
      console.error('[BedrockAPI] 无法找到对应的管理工具调用:', toolCall.toolUseId)
      return false
    }

    // 标记工具调用为失败状态
    state.markToolCallAsFailed(managedTool, error)

    console.warn(
      `[BedrockAPI] 工具调用失败: ${toolCall.name}, 错误: ${error}, 重试次数: ${managedTool.retryCount}/${state.maxRetryCount}`
    )

    // 如果还可以重试，返回 true
    return managedTool.state === ToolCallState.FAILED
  }

  /**
   * 检查是否需要停止整个工具调用流程
   * @param state - 流状态管理器
   * @returns 是否应该停止
   */
  public shouldStopToolCallProcess(state: StreamState): boolean {
    const cancelledTools = state.getCancelledToolCalls()
    return cancelledTools.length > 0 || state.shouldStopRetrying()
  }

  /**
   * 类型判断：是否为Converse命令输出
   * @param output - 输出数据
   * @returns 类型判断结果
   */
  private isConverseCommandOutput(output: BedrockSdkRawOutput | string | undefined): output is ConverseCommandOutput {
    return typeof output === 'object' && output !== null && 'output' in output && 'stopReason' in output
  }

  /**
   * 构建推理配置
   * @param assistant - 助手配置
   * @param model - 模型信息
   * @returns 推理配置或undefined
   */
  private buildReasoningConfig(assistant: Assistant, model: Model): Record<string, any> | undefined {
    if (!isReasoningModel(model)) return undefined

    const { maxTokens } = getAssistantSettings(assistant)
    const reasoningEffort = assistant?.settings?.reasoning_effort

    if (reasoningEffort === undefined) {
      return { thinking: { type: 'disabled' } }
    }

    const budgetTokens = this.calculateThinkingBudget(model, reasoningEffort, maxTokens)
    return {
      thinking: {
        type: 'enabled',
        budget_tokens: budgetTokens
      }
    }
  }

  /**
   * 计算思考预算令牌数
   * @param model - 模型信息
   * @param reasoningEffort - 推理努力程度
   * @param maxTokens - 最大令牌数
   * @returns 预算令牌数
   */
  private calculateThinkingBudget(model: Model, reasoningEffort: string, maxTokens?: number): number {
    const effortRatio = EFFORT_RATIO[reasoningEffort] || 0.5
    const tokenLimit = findTokenLimit(model.id)

    if (!tokenLimit) return BedrockAPIClient.MIN_THINKING_BUDGET_TOKENS

    const dynamicBudget = (tokenLimit.max - tokenLimit.min) * effortRatio + tokenLimit.min
    const maxAllowedBudget = (maxTokens || DEFAULT_MAX_TOKENS) * effortRatio

    return Math.max(BedrockAPIClient.MIN_THINKING_BUDGET_TOKENS, Math.floor(Math.min(dynamicBudget, maxAllowedBudget)))
  }

  /**
   * 转换消息格式为Bedrock格式
   * @param messages - 消息列表
   * @returns 转换后的消息列表
   */
  private transformMessagesForBedrock(messages: BedrockSdkMessageParam[]) {
    return messages.map((message) => ({
      role: message.role as ConversationRole,
      content: message.content as ContentBlock[]
    }))
  }

  /**
   * 转换单个消息为SDK参数格式
   * @param message - 消息对象
   * @param model - 模型信息
   * @returns SDK消息参数
   */
  public async convertMessageToSdkParam(message: Message, model: Model): Promise<BedrockSdkMessageParam> {
    const isVisionCapable = isVisionModel(model)
    const messageContent = await this.getMessageContent(message)
    const contentBlocks: ContentBlock[] = []

    if (messageContent) {
      contentBlocks.push({ text: messageContent })
    }

    await this.processImageContent(message, contentBlocks, isVisionCapable)
    await this.processFileContent(message, contentBlocks)

    return {
      role: message.role === 'system' ? 'user' : message.role,
      content: contentBlocks
    } as BedrockSdkMessageParam
  }

  /**
   * 处理图像内容
   * @param message - 消息对象
   * @param contentBlocks - 内容块列表
   * @param isVisionCapable - 是否支持视觉
   */
  private async processImageContent(
    message: Message,
    contentBlocks: ContentBlock[],
    isVisionCapable: boolean
  ): Promise<void> {
    if (!isVisionCapable) return

    const imageBlocks = findImageBlocks(message)
    for (const imageBlock of imageBlocks) {
      if (imageBlock.file) {
        const imageContentBlock = await this.convertImageFileToContentBlock(imageBlock.file)
        if (imageContentBlock) {
          contentBlocks.push(imageContentBlock)
        }
      }
    }
  }

  /**
   * 转换图像文件为内容块
   * @param file - 文件对象
   * @returns 内容块或null
   */
  private async convertImageFileToContentBlock(file: any): Promise<ContentBlock | null> {
    try {
      const imageData = await window.api.file.base64Image(file.id + file.ext)
      const base64Content = imageData.data.split(',')[1]
      const imageFormat = imageData.data.includes('jpeg') ? 'jpeg' : 'png'

      return {
        image: {
          format: imageFormat as 'jpeg' | 'png' | 'gif' | 'webp',
          source: {
            bytes: new Uint8Array(Buffer.from(base64Content, 'base64'))
          }
        }
      }
    } catch (error) {
      console.error('[BedrockAPI] 转换图像文件失败:', error)
      return null
    }
  }

  /**
   * 处理文件内容
   * @param message - 消息对象
   * @param contentBlocks - 内容块列表
   */
  private async processFileContent(message: Message, contentBlocks: ContentBlock[]): Promise<void> {
    const fileBlocks = findFileBlocks(message)
    for (const fileBlock of fileBlocks) {
      const file = fileBlock.file
      if (file && [FileTypes.TEXT, FileTypes.DOCUMENT].includes(file.type)) {
        try {
          const fileContent = await window.api.file.read(file.id + file.ext)
          contentBlocks.push({
            text: `${file.origin_name}\n${fileContent.trim()}`
          })
        } catch (error) {
          console.error('[BedrockAPI] 读取文件内容失败:', error)
        }
      }
    }
  }

  /**
   * 转换MCP工具为SDK工具格式
   * @param mcpTools - MCP工具列表
   * @returns SDK工具列表
   */
  convertMcpToolsToSdkTools(mcpTools: MCPTool[]): BedrockSdkTool[] {
    return mcpTools.map(
      (tool) =>
        ({
          toolSpec: {
            name: tool.name,
            description: tool.description,
            inputSchema: {
              json: tool.inputSchema as any
            }
          }
        }) as BedrockSdkTool
    )
  }

  /**
   * 根据工具调用查找对应的MCP工具
   * @param toolCall - 工具调用
   * @param mcpTools - MCP工具列表
   * @returns MCP工具或undefined
   */
  convertSdkToolCallToMcp(toolCall: BedrockSdkToolCall, mcpTools: MCPTool[]): MCPTool | undefined {
    return mcpTools.find((tool) => tool.name === toolCall.name)
  }

  /**
   * 转换SDK工具调用为MCP工具响应
   * @param toolCall - SDK工具调用
   * @param mcpTool - MCP工具
   * @returns 工具调用响应
   */
  convertSdkToolCallToMcpToolResponse(toolCall: BedrockSdkToolCall, mcpTool: MCPTool): ToolCallResponse {
    let parsedArguments: any = {}

    if (typeof toolCall.input === 'string') {
      try {
        if (toolCall.input.trim()) {
          parsedArguments = JSON.parse(toolCall.input)
        }
      } catch (error) {
        console.error('[BedrockAPI] 解析工具调用参数失败:', toolCall.input, error)
        parsedArguments = {}
      }
    } else if (toolCall.input && typeof toolCall.input === 'object') {
      parsedArguments = toolCall.input
    }

    return {
      id: toolCall.toolUseId,
      toolCallId: toolCall.toolUseId,
      tool: mcpTool,
      arguments: parsedArguments,
      status: 'pending'
    } as ToolCallResponse
  }

  /**
   * 转换MCP工具响应为SDK消息参数
   * @param mcpToolResponse - MCP工具响应
   * @param response - 调用响应
   * @returns SDK消息参数或undefined
   */
  convertMcpToolResponseToSdkMessageParam(
    mcpToolResponse: MCPToolResponse,
    response: MCPCallToolResponse
  ): BedrockSdkMessageParam | undefined {
    const toolUseId = this.extractToolUseId(mcpToolResponse)
    if (!toolUseId) {
      console.error('[BedrockAPI] 无法提取工具使用ID')
      return undefined
    }

    const resultText = this.extractResponseText(response)

    return {
      role: 'user',
      content: [
        {
          toolResult: {
            toolUseId,
            content: [{ text: resultText }]
          }
        }
      ]
    } as BedrockSdkMessageParam
  }

  /**
   * 提取工具使用ID
   * @param mcpToolResponse - MCP工具响应
   * @returns 工具使用ID或undefined
   */
  private extractToolUseId(mcpToolResponse: MCPToolResponse): string | undefined {
    if ('toolUseId' in mcpToolResponse && mcpToolResponse.toolUseId) {
      return mcpToolResponse.toolUseId
    }
    if ('toolCallId' in mcpToolResponse && mcpToolResponse.toolCallId) {
      return mcpToolResponse.toolCallId
    }
    return undefined
  }

  /**
   * 提取响应文本内容
   * @param response - 调用响应
   * @returns 响应文本
   */
  private extractResponseText(response: MCPCallToolResponse): string {
    if (Array.isArray(response.content) && response.content.length > 0 && response.content[0].text) {
      return response.content.map((content) => content.text || '').join('\n')
    }
    if (typeof response.content === 'object') {
      return JSON.stringify(response.content)
    }
    return String(response.content)
  }

  /**
   * 构建SDK消息列表
   * @param currentMessages - 当前消息列表
   * @param output - 输出数据
   * @param toolResults - 工具结果列表
   * @param toolCalls - 工具调用列表
   * @returns SDK消息列表
   */
  override buildSdkMessages(
    currentMessages: BedrockSdkMessageParam[],
    output: BedrockSdkRawOutput | string | undefined,
    toolResults: BedrockSdkMessageParam[],
    toolCalls?: BedrockSdkToolCall[]
  ): BedrockSdkMessageParam[] {
    const messages = [...currentMessages]

    if (this.isConverseCommandOutput(output)) {
      const assistantContent = output.output?.message?.content
      if (assistantContent) {
        messages.push({
          role: 'assistant',
          content: assistantContent
        })
      }
    } else {
      const assistantContent: ContentBlock[] = []
      const hasToolCalls = toolCalls && toolCalls.length > 0

      if (hasToolCalls) {
        const thinkingText = this.lastThinkingBlock?.thinking || (typeof output === 'string' && output ? output : '')

        if (thinkingText && thinkingText.trim()) {
          assistantContent.push({
            reasoningContent: {
              reasoningText: {
                text: thinkingText,
                signature: this.lastThinkingBlock?.signature
              }
            }
          })
        }

        for (const toolCall of toolCalls) {
          let parsedInput: any
          try {
            parsedInput = typeof toolCall.input === 'string' ? JSON.parse(toolCall.input) : toolCall.input
          } catch (error) {
            console.error('[BedrockAPI] Failed to parse tool input:', toolCall.input, error)
            parsedInput = {}
          }

          assistantContent.push({
            toolUse: {
              toolUseId: toolCall.toolUseId,
              name: toolCall.name,
              input: parsedInput || {}
            }
          })
        }
      } else if (typeof output === 'string' && output) {
        assistantContent.push({ text: output })
      }

      if (assistantContent.length > 0) {
        messages.push({
          role: 'assistant',
          content: assistantContent
        })
      }
    }

    if (toolResults && toolResults.length > 0) {
      messages.push(...toolResults)
    }

    return messages
  }

  /**
   * 估算消息令牌数
   * @param message - 消息对象
   * @returns 令牌数量
   */
  override estimateMessageTokens(message: BedrockSdkMessageParam): number {
    let tokenCount = 0

    if (message.content && Array.isArray(message.content)) {
      for (const contentBlock of message.content) {
        if ('text' in contentBlock && contentBlock.text) {
          tokenCount += estimateTextTokens(contentBlock.text)
        }
      }
    }

    return tokenCount
  }

  /**
   * 从SDK载荷中提取消息
   * @param sdkPayload - SDK载荷
   * @returns 消息列表
   */
  extractMessagesFromSdkPayload(sdkPayload: BedrockSdkParams): BedrockSdkMessageParam[] {
    return sdkPayload.messages || []
  }

  /**
   * 获取请求转换器
   * @returns 请求转换器
   */
  getRequestTransformer(): RequestTransformer<BedrockSdkParams, BedrockSdkMessageParam> {
    return {
      transform: async (coreRequest, assistant, model, isRecursiveCall, recursiveSdkMessages) => {
        const { messages, mcpTools, maxTokens, streamOutput } = coreRequest

        this.setupToolsConfig({ mcpTools, model, enableToolUse: true })
        const tools = this.useSystemPromptForTools ? [] : mcpTools ? this.convertMcpToolsToSdkTools(mcpTools) : []

        let systemContent = assistant.prompt || ''
        if (this.useSystemPromptForTools && mcpTools) {
          systemContent = await buildSystemPrompt(systemContent, mcpTools, assistant)
        }

        const userMessages = await this.processUserMessages(messages, model)
        const requestMessages = isRecursiveCall && recursiveSdkMessages?.length ? recursiveSdkMessages : userMessages

        const inferenceConfig = this.buildInferenceConfiguration(assistant, model, maxTokens)
        const reasoningConfig = this.buildReasoningConfig(assistant, model)

        const sdkParams: BedrockSdkParams = {
          modelId: this.getModelId(model),
          messages: requestMessages,
          system: systemContent ? [{ text: systemContent }] : undefined,
          inferenceConfig,
          toolConfig: tools.length > 0 ? ({ tools } as ToolConfiguration) : undefined,
          additionalModelRequestFields: reasoningConfig,
          stream: streamOutput
        }

        const timeout = this.getTimeout(model)
        return {
          payload: sdkParams,
          messages: requestMessages,
          metadata: { timeout }
        }
      }
    }
  }

  /**
   * 处理用户消息
   * @param messages - 消息或消息列表
   * @param model - 模型信息
   * @returns SDK消息参数列表
   */
  private async processUserMessages(messages: string | Message[], model: Model): Promise<BedrockSdkMessageParam[]> {
    const userMessages: BedrockSdkMessageParam[] = []

    if (typeof messages === 'string') {
      userMessages.push({
        role: 'user',
        content: [{ text: messages }]
      })
    } else {
      const processedMessages = addImageFileToContents(messages)
      for (const message of processedMessages) {
        userMessages.push(await this.convertMessageToSdkParam(message, model))
      }
    }

    return userMessages
  }

  /**
   * 构建推理配置
   * @param assistant - 助手配置
   * @param model - 模型信息
   * @param maxTokens - 最大令牌数
   * @returns 推理配置
   */
  private buildInferenceConfiguration(assistant: Assistant, model: Model, maxTokens?: number): InferenceConfiguration {
    return {
      maxTokens: maxTokens || DEFAULT_MAX_TOKENS,
      temperature: this.getTemperature(assistant, model),
      topP: this.getTopP(assistant, model)
    }
  }

  /**
   * 获取响应块转换器
   * @returns 响应块转换器
   */
  getResponseChunkTransformer(): ResponseChunkTransformer<BedrockSdkRawChunk> {
    return () => {
      const streamState = new StreamState()
      const streamProcessor = new StreamProcessor(this)

      return {
        transform: (chunk: BedrockSdkRawChunk, controller: TransformStreamDefaultController<GenericChunk>) => {
          streamProcessor.processEvent(chunk, streamState, controller)
        }
      }
    }
  }
}
