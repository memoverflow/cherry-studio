/* eslint-disable @typescript-eslint/no-unused-vars */
import {
  BedrockRuntimeClient,
  CitationsDelta,
  ContentBlock as BedrockContentBlock,
  ContentBlock,
  ContentBlockDelta,
  ContentBlockDeltaEvent,
  ContentBlockStart,
  ContentBlockStartEvent,
  ContentBlockStopEvent,
  ConversationRole,
  ConverseCommand,
  ConverseCommandOutput,
  ConverseResponse,
  ConverseStreamCommand,
  ConverseStreamMetadataEvent,
  InferenceConfiguration,
  MessageStopEvent,
  ReasoningContentBlockDelta,
  ReasoningTextBlock,
  TokenUsage,
  ToolConfiguration,
  ToolUseBlockDelta
} from '@aws-sdk/client-bedrock-runtime'
import { DEFAULT_MAX_TOKENS } from '@renderer/config/constant'
import { findTokenLimit, isReasoningModel, isVisionModel } from '@renderer/config/models'
import { getAssistantSettings } from '@renderer/services/AssistantService'
import { estimateTextTokens } from '@renderer/services/TokenService'
import {
  Assistant,
  EFFORT_RATIO,
  FileTypes,
  GenerateImageParams,
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
 * Bedrock流式响应事件类型枚举
 */
enum BedrockStreamEventType {
  MESSAGE_START = 'messageStart',
  CONTENT_BLOCK_START = 'contentBlockStart',
  CONTENT_BLOCK_DELTA = 'contentBlockDelta',
  CONTENT_BLOCK_STOP = 'contentBlockStop',
  MESSAGE_STOP = 'messageStop',
  METADATA = 'metadata'
}

/**
 * Bedrock 流式事件的包装器接口
 * 实际的事件数据都是嵌套在对应的属性中
 */
interface BedrockStreamEventWrapper {
  messageStart?: any
  contentBlockStart?: ContentBlockStartEvent
  contentBlockDelta?: ContentBlockDeltaEvent
  contentBlockStop?: ContentBlockStopEvent
  messageStop?: any
  metadata?: any
}

/**
 * Bedrock流处理状态管理器
 * 用于跟踪流式响应的处理状态，包括文本输出、思考内容和工具调用
 */
class BedrockStreamState {
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

  /** 工具调用列表 */
  toolCalls: BedrockSdkToolCall[] = []

  /** 累积的思考内容 */
  thinkingContent = ''

  /** 思考签名 */
  thinkingSignature = ''

  /** 是否有工具调用已发送 */
  hasToolCallsSent = false

  /** 累积的token使用量 */
  accumulatedUsage: {
    inputTokens: number
    outputTokens: number
    totalTokens: number
  } = {
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0
  }

  /**
   * 重置状态到初始值
   */
  reset(): void {
    this.hasStartedText = false
    this.hasStartedThinking = false
    this.thinkingStartTime = undefined
    this.stopReason = undefined
    this.usage = undefined
    this.toolCalls = []
    this.thinkingContent = ''
    this.thinkingSignature = ''
    this.hasToolCallsSent = false
    // 注意：不重置accumulatedUsage，因为需要在整个对话过程中累加
  }

  /**
   * 累加token使用量
   */
  addUsage(usage: TokenUsage): void {
    this.accumulatedUsage.inputTokens += usage.inputTokens || 0
    this.accumulatedUsage.outputTokens += usage.outputTokens || 0
    this.accumulatedUsage.totalTokens += usage.totalTokens || 0
  }

  /**
   * 重置累积的token使用量
   */
  resetAccumulatedUsage(): void {
    this.accumulatedUsage = {
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0
    }
  }
}

/**
 * 流式响应处理器基础接口
 * 定义了处理Bedrock流式响应事件的标准接口
 */
interface BedrockStreamEventHandler {
  /** 判断是否可以处理指定的事件 */
  canHandle(event: BedrockSdkRawChunk): boolean

  /** 处理事件并返回是否成功处理 */
  handle(
    event: BedrockSdkRawChunk,
    state: BedrockStreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): boolean
}

/**
 * 消息开始事件处理器
 * 处理流式响应的开始事件，重置处理状态
 */
class MessageStartEventHandler implements BedrockStreamEventHandler {
  canHandle(event: BedrockSdkRawChunk): boolean {
    return 'messageStart' in event
  }

  handle(
    event: BedrockSdkRawChunk,
    state: BedrockStreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): boolean {
    // 类型安全地访问嵌套的 messageStart 属性
    const eventWrapper = event as BedrockStreamEventWrapper
    const messageStart = eventWrapper.messageStart
    if (!messageStart) {
      return false
    }

    // 消息开始时重置所有状态
    state.reset()

    // 不在这里发送TEXT_START，等到真正有文本内容时再发送

    return true
  }
}

/**
 * 内容块开始事件处理器
 * 处理工具调用开始事件，初始化工具调用状态
 */
class ContentBlockStartEventHandler implements BedrockStreamEventHandler {
  canHandle(event: BedrockSdkRawChunk): boolean {
    return BedrockStreamEventType.CONTENT_BLOCK_START in event
  }

  handle(
    event: BedrockSdkRawChunk,
    state: BedrockStreamState,
    _controller: TransformStreamDefaultController<GenericChunk>
  ): boolean {
    // 类型安全地访问嵌套的 contentBlockStart 属性
    const eventWrapper = event as BedrockStreamEventWrapper
    const startEvent = eventWrapper.contentBlockStart
    if (!startEvent?.start) {
      return false
    }

    return ContentBlockStart.visit(startEvent.start, {
      toolUse: (toolUseStart) => {
        try {
          state.toolCalls.push({
            toolUseId: toolUseStart.toolUseId || `tool_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`,
            name: toolUseStart.name || 'unknown_tool',
            input: '',
            type: 'tool_use'
          })
          return true
        } catch (error) {
          console.error('[BedrockAPI] 处理工具调用开始事件失败:', error)
          return false
        }
      },
      _: () => false
    })
  }
}

/**
 * 内容块增量事件处理器
 * 处理流式内容增量更新，包括文本、思考内容和工具输入
 */
class ContentBlockDeltaEventHandler implements BedrockStreamEventHandler {
  canHandle(event: BedrockSdkRawChunk): boolean {
    return BedrockStreamEventType.CONTENT_BLOCK_DELTA in event
  }

  handle(
    event: BedrockSdkRawChunk,
    state: BedrockStreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): boolean {
    // 类型安全地访问嵌套的 contentBlockDelta 属性
    const eventWrapper = event as BedrockStreamEventWrapper
    const deltaEvent = eventWrapper.contentBlockDelta
    if (!deltaEvent?.delta) {
      return false
    }

    console.log(`[BedrockAPI] ContentBlockDelta - Index: ${deltaEvent.contentBlockIndex}, Delta:`, deltaEvent.delta)

    // 使用AWS SDK的visitor模式优雅处理联合类型
    return ContentBlockDelta.visit(deltaEvent.delta, {
      text: (text) => {
        // contentBlockIndex 0 表示非thinking模式下的文本内容或thinking模式下的普通文本
        // contentBlockIndex 1 表示thinking模式下thinking块后的文本内容
        // 所以我们需要处理所有的文本内容，不管contentBlockIndex是什么
        this.handleTextContent(text, state, controller)
        return true
      },
      reasoningContent: (reasoningDelta: ReasoningContentBlockDelta) => {
        // contentBlockIndex 0 表示 thinking 内容
        let handled = false
        if ('text' in reasoningDelta && reasoningDelta.text) {
          this.handleReasoningContent(reasoningDelta.text, state, controller)
          handled = true
        }
        if ('signature' in reasoningDelta && reasoningDelta.signature) {
          state.thinkingSignature += reasoningDelta.signature
          handled = true
        }
        return handled
      },
      toolUse: (toolUseDelta: ToolUseBlockDelta) => {
        console.log('toolUseDelta', toolUseDelta)
        console.log('state.toolCalls', state.toolCalls)

        if (toolUseDelta.input === undefined || state.toolCalls.length === 0) {
          return false
        }

        this.handleToolInputDelta(toolUseDelta.input, state)
        this.tryToSendToolCallIfComplete(state, controller)
        return true
      },
      citation: (_citation: CitationsDelta) => {
        // 暂不处理引用内容
        return false
      },
      _: () => false
    })
  }

  /**
   * 处理文本内容增量
   */
  private handleTextContent(
    text: string,
    state: BedrockStreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): void {
    // 如果是第一次处理文本，先发送TEXT_START
    if (!state.hasStartedText) {
      controller.enqueue({ type: ChunkType.TEXT_START } as TextStartChunk)
      state.hasStartedText = true
    }
    console.log('[BedrockAPI] handleTextContent:', text)
    controller.enqueue({ type: ChunkType.TEXT_DELTA, text })
  }

  /**
   * 处理推理思考内容增量
   */
  private handleReasoningContent(
    text: string,
    state: BedrockStreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): void {
    if (!state.hasStartedThinking) {
      controller.enqueue({ type: ChunkType.THINKING_START } as ThinkingStartChunk)
      state.hasStartedThinking = true
      state.thinkingContent = ''
      state.thinkingStartTime = Date.now()
    }

    state.thinkingContent += text
    controller.enqueue({
      type: ChunkType.THINKING_DELTA,
      text
    } as ThinkingDeltaChunk)
  }

  /**
   * 处理工具输入增量
   */
  private handleToolInputDelta(inputDelta: string, state: BedrockStreamState): void {
    const currentToolCall = state.toolCalls[state.toolCalls.length - 1]
    if (currentToolCall) {
      currentToolCall.input += inputDelta
      console.log('currentToolCall.input', currentToolCall.input)
      console.log('[BedrockAPI] Tool input delta added:', {
        toolName: currentToolCall.name,
        delta: inputDelta,
        currentInput: currentToolCall.input,
        inputLength: currentToolCall.input.length
      })
    }
  }

  /**
   * 检查并发送完整的工具调用
   */
  private tryToSendToolCallIfComplete(
    state: BedrockStreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): void {
    const currentToolCall = state.toolCalls[state.toolCalls.length - 1] as BedrockSdkToolCall & { sent?: boolean }
    if (!currentToolCall || currentToolCall.sent) {
      return
    }

    const [isComplete, parsedInput] = this.isToolInputComplete(currentToolCall.input)
    if (isComplete) {
      console.log('[BedrockAPI] Tool input complete, sending tool call:', currentToolCall)
      console.log('[BedrockAPI] currentToolCall keys:', Object.keys(currentToolCall))
      console.log('[BedrockAPI] currentToolCall descriptor:', Object.getOwnPropertyDescriptor(currentToolCall, 'input'))
      console.log('[BedrockAPI] Direct input access:', currentToolCall.input)
      console.log('[BedrockAPI] Bracket input access:', currentToolCall['input'])
      console.log('[BedrockAPI] JSON stringify:', JSON.stringify(currentToolCall))

      // 不要修改原始的input，让convertSdkToolCallToMcpToolResponse来处理转换
      controller.enqueue({
        type: ChunkType.MCP_TOOL_CREATED,
        tool_calls: [currentToolCall]
      })
      currentToolCall.sent = true
    }
  }

  /**
   * 检查工具输入是否完整
   */
  private isToolInputComplete(input: string): [boolean, any] {
    // 空字符串表示无参数工具调用
    if (input === '') {
      console.log('[BedrockAPI] Empty tool input detected (no parameters)')
      return [true, '']
    }

    // 非空字符串需要是有效的JSON
    if (input.trim()) {
      try {
        const inputJson = JSON.parse(input)
        console.log('inputJson', inputJson)
        console.log('[BedrockAPI] Valid JSON input detected')
        return [true, inputJson]
      } catch (error) {
        console.log('[BedrockAPI] Tool input not yet complete, continuing to accumulate')
        return [false, '']
      }
    }

    return [false, '']
  }
}

/**
 * 内容块停止事件处理器
 * 处理内容块结束事件，包括thinking结束
 */
class ContentBlockStopEventHandler implements BedrockStreamEventHandler {
  canHandle(event: BedrockSdkRawChunk): boolean {
    return BedrockStreamEventType.CONTENT_BLOCK_STOP in event
  }

  handle(
    event: BedrockSdkRawChunk,
    state: BedrockStreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): boolean {
    // 类型安全地访问嵌套的 contentBlockStop 属性
    const eventWrapper = event as BedrockStreamEventWrapper
    const stopEvent = eventWrapper.contentBlockStop
    if (!stopEvent) {
      return false
    }

    console.log(`[BedrockAPI] ContentBlockStop - Index: ${stopEvent.contentBlockIndex}`)

    // contentBlockIndex 0 表示 thinking 结束
    // 发送TEXT_COMPLETE来触发中间件检测thinking结束
    if (stopEvent.contentBlockIndex === 0 && state.hasStartedThinking) {
      console.log(
        `[BedrockAPI] Thinking block ${stopEvent.contentBlockIndex} ended, sending TEXT_COMPLETE to trigger middleware`
      )
      controller.enqueue({
        type: ChunkType.TEXT_COMPLETE,
        text: ''
      } as TextCompleteChunk)
    }

    return true
  }
}

/**
 * 消息停止事件处理器
 * 处理消息结束事件，记录停止原因
 */
class MessageStopEventHandler implements BedrockStreamEventHandler {
  canHandle(event: BedrockSdkRawChunk): boolean {
    return BedrockStreamEventType.MESSAGE_STOP in event
  }

  handle(
    event: BedrockSdkRawChunk,
    state: BedrockStreamState,
    _controller: TransformStreamDefaultController<GenericChunk>
  ): boolean {
    const stopEvent = event as MessageStopEvent
    state.stopReason = stopEvent.stopReason
    return true
  }
}

/**
 * 元数据事件处理器
 * 处理响应完成后的元数据，包括使用统计和最终状态处理
 */
class MetadataEventHandler implements BedrockStreamEventHandler {
  private client: BedrockAPIClient

  constructor(client: BedrockAPIClient) {
    this.client = client
  }

  canHandle(event: BedrockSdkRawChunk): boolean {
    return BedrockStreamEventType.METADATA in event && 'metadata' in event && (event as any).metadata?.usage
  }

  handle(
    event: BedrockSdkRawChunk,
    state: BedrockStreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): boolean {
    const metadataEvent = (event as any).metadata as ConverseStreamMetadataEvent

    console.log('metadataEvent', metadataEvent)
    state.usage = metadataEvent.usage

    // 累加token使用量到全局累积中
    if (state.usage) {
      this.client.addToGlobalUsage(state.usage)
      state.addUsage(state.usage)
      console.log('[BedrockAPI] Local accumulated usage:', state.accumulatedUsage)
    }

    // 保存完整的思考块用于后续工具调用
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
   * 完成响应处理
   */
  private finalizeResponse(
    state: BedrockStreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): void {
    if (state.stopReason === 'tool_use' && state.toolCalls.length > 0) {
      // 如果还没有发送过工具调用，现在发送所有工具调用
      if (!state.hasToolCallsSent) {
        const processedToolCalls = this.client.processToolCalls(state.toolCalls)
        console.log('[BedrockAPI] Processing all tool calls (stream):', processedToolCalls)
        controller.enqueue({
          type: ChunkType.MCP_TOOL_CREATED,
          tool_calls: processedToolCalls
        })
        state.hasToolCallsSent = true
      }
      // 不要在这里关闭流，让工具执行完成后由中间件继续处理
    } else {
      // 发送全局累积的使用统计
      const globalUsage = this.client.getGlobalUsage()
      if (globalUsage.totalTokens > 0) {
        this.emitUsageStatistics(controller, globalUsage)
      }
    }
  }

  /**
   * 发送使用统计信息
   */
  private emitUsageStatistics(
    controller: TransformStreamDefaultController<GenericChunk>,
    usage: TokenUsage | { inputTokens: number; outputTokens: number; totalTokens: number }
  ): void {
    controller.enqueue({
      type: ChunkType.LLM_RESPONSE_COMPLETE,
      response: {
        usage: {
          prompt_tokens: usage.inputTokens || 0,
          completion_tokens: usage.outputTokens || 0,
          total_tokens: usage.totalTokens || 0
        }
      }
    })
  }
}

/**
 * 非流式响应处理器
 * 处理同步响应，将其转换为流式格式输出
 */
class NonStreamResponseHandler implements BedrockStreamEventHandler {
  private client: BedrockAPIClient

  constructor(client: BedrockAPIClient) {
    this.client = client
  }
  canHandle(event: BedrockSdkRawChunk): boolean {
    return 'output' in event && 'stopReason' in event && (event as any).output
  }

  handle(
    event: BedrockSdkRawChunk,
    state: BedrockStreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): boolean {
    const converseResponse = event as ConverseResponse
    console.log('non-stream-event1', event)
    console.log('non-stream-event2', converseResponse)
    console.log('non-stream-event3', converseResponse.stopReason)
    if (converseResponse.usage) {
      state.usage = converseResponse.usage
    }

    const messageContent = converseResponse.output?.message?.content
    if (!messageContent || !Array.isArray(messageContent)) {
      return false
    }

    let hasProcessedContent = false

    // 使用AWS SDK的visitor模式处理内容块
    messageContent.forEach((contentBlock) => {
      ContentBlock.visit(contentBlock, {
        text: (text) => {
          this.handleTextBlock(text, state, controller)
          hasProcessedContent = true
        },
        toolUse: (toolUseBlock) => {
          this.handleToolUseBlock(toolUseBlock, state)
          hasProcessedContent = true
        },
        image: () => {
          // 暂不处理图像内容
        },
        document: () => {
          // 暂不处理文档内容
        },
        video: () => {
          // 暂不处理视频内容
        },
        toolResult: () => {
          // 暂不处理工具结果
        },
        guardContent: () => {
          // 暂不处理守护内容
        },
        cachePoint: () => {
          // 暂不处理缓存点
        },
        reasoningContent: () => {
          // 暂不处理推理内容
        },
        citationsContent: () => {
          // 暂不处理引用内容
        },
        _: () => {
          // 未知类型暂不处理
        }
      })
    })

    if (hasProcessedContent) {
      state.stopReason = converseResponse.stopReason
      this.finalizeNonStreamResponse(state, controller)
    }

    return hasProcessedContent
  }

  /**
   * 处理文本块
   */
  private handleTextBlock(
    text: string,
    state: BedrockStreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): void {
    if (!state.hasStartedText) {
      controller.enqueue({ type: ChunkType.TEXT_START } as TextStartChunk)
      state.hasStartedText = true
    }
    controller.enqueue({ type: ChunkType.TEXT_DELTA, text })
  }

  /**
   * 处理工具使用块
   */
  private handleToolUseBlock(
    toolUseBlock: { toolUseId: string | undefined; name: string | undefined; input: any },
    state: BedrockStreamState
  ): void {
    state.toolCalls.push({
      toolUseId: toolUseBlock.toolUseId || `tool_${Date.now()}`,
      name: toolUseBlock.name || 'unknown_tool',
      input: toolUseBlock.input || {},
      type: 'tool_use'
    })
  }

  /**
   * 完成非流式响应处理
   */
  private finalizeNonStreamResponse(
    state: BedrockStreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): void {
    if (state.stopReason === 'tool_use' && state.toolCalls.length > 0) {
      // 处理工具调用
      const processedToolCalls = this.client.processToolCalls(state.toolCalls)
      console.log('[BedrockAPI] Processing tool calls (non-stream):', processedToolCalls)
      controller.enqueue({
        type: ChunkType.MCP_TOOL_CREATED,
        tool_calls: processedToolCalls
      })
      // 不要在这里关闭流，让工具执行完成后由中间件继续处理
    } else if (state.usage) {
      // 累加token使用量到全局累积中
      this.client.addToGlobalUsage(state.usage)
      state.addUsage(state.usage)
      console.log('[BedrockAPI] Local accumulated usage (non-stream):', state.accumulatedUsage)

      // 发送全局累积的使用统计
      const globalUsage = this.client.getGlobalUsage()
      controller.enqueue({
        type: ChunkType.LLM_RESPONSE_COMPLETE,
        response: {
          usage: {
            prompt_tokens: globalUsage.inputTokens,
            completion_tokens: globalUsage.outputTokens,
            total_tokens: globalUsage.totalTokens
          }
        }
      })
    }
  }
}

/**
 * Bedrock流式响应处理器
 * 统一管理所有类型的流式响应事件处理
 */
class BedrockStreamProcessor {
  private eventHandlers: BedrockStreamEventHandler[]

  constructor(client: BedrockAPIClient) {
    this.eventHandlers = [
      new MessageStartEventHandler(),
      new ContentBlockStartEventHandler(),
      new ContentBlockDeltaEventHandler(),
      new ContentBlockStopEventHandler(),
      new MessageStopEventHandler(),
      new MetadataEventHandler(client),
      new NonStreamResponseHandler(client)
    ]
  }

  /**
   * 处理单个流式响应事件
   */
  processEvent(
    event: BedrockSdkRawChunk,
    state: BedrockStreamState,
    controller: TransformStreamDefaultController<GenericChunk>
  ): boolean {
    if (!event) {
      console.warn('[BedrockAPI] 接收到空的流式事件')
      return false
    }

    for (const handler of this.eventHandlers) {
      if (handler.canHandle(event)) {
        try {
          return handler.handle(event, state, controller)
        } catch (error) {
          console.error('[BedrockAPI] 处理流式事件失败:', error)
          return false
        }
      }
    }

    console.warn('[BedrockAPI] 未找到适合的事件处理器:', event)
    return false
  }
}

/**
 * Amazon Bedrock API 客户端
 *
 * 这是一个完全重构的Bedrock API客户端，充分利用了AWS SDK的原生类和接口：
 * - 使用BedrockRuntimeClient作为底层客户端
 * - 支持ConverseCommand和ConverseStreamCommand进行对话
 * - 实现了完整的工具调用和流式响应处理
 * - 提供了推理模型的思考模式支持
 * - 集成了Cherry Studio的业务逻辑和类型系统
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

  /** 最近一次响应的思考块数据，用于递归工具调用 */
  private lastThinkingBlock: { thinking: string; signature: string } | null = null

  /** 全局累积的token使用量，在整个会话中保持 */
  private globalAccumulatedUsage: {
    inputTokens: number
    outputTokens: number
    totalTokens: number
  } = {
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0
  }

  /** 全局会话状态，在多次请求之间保持某些状态 */
  private sessionState: {
    hasEverStartedText: boolean
    conversationDepth: number
  } = {
    hasEverStartedText: false,
    conversationDepth: 0
  }

  constructor(provider: Provider) {
    super(provider)
  }

  /**
   * 生成图像功能
   * 注意：当前Bedrock SDK暂不支持图像生成，返回空数组
   */
  override async generateImage(_params: GenerateImageParams): Promise<string[]> {
    console.warn('[BedrockAPI] Bedrock SDK暂不支持图像生成功能')
    return []
  }

  /**
   * 获取嵌入向量维度
   * 注意：当前Bedrock SDK暂不支持嵌入向量，抛出错误
   */
  override async getEmbeddingDimensions(): Promise<number> {
    throw new Error('[BedrockAPI] Bedrock SDK暂不支持嵌入向量功能')
  }

  /**
   * 列出可用模型
   * 注意：当前实现返回空数组，可根据需要扩展
   */
  override async listModels(): Promise<SdkModel[]> {
    console.warn('[BedrockAPI] 模型列表功能尚未实现')
    return []
  }

  /**
   * 创建聊天完成请求
   * 这是核心方法，支持流式和非流式响应
   */
  override async createCompletions(payload: BedrockSdkParams): Promise<BedrockSdkRawOutput> {
    const client = await this.getSdkInstance()
    const messages = this.transformMessagesForBedrock(payload.messages)

    // 构建命令参数
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
        // 流式响应
        const command = new ConverseStreamCommand(commandParams)
        const response = await client.send(command)
        return response.stream as BedrockSdkRawOutput
      } else {
        // 非流式响应
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
   * 获取Bedrock SDK实例
   * 实现懒加载，确保客户端配置正确
   */
  async getSdkInstance(): Promise<BedrockSdkInstance> {
    if (!this.bedrockClient) {
      this.bedrockClient = this.createBedrockRuntimeClient()
    }
    return this.bedrockClient as BedrockSdkInstance
  }

  /**
   * 创建Bedrock Runtime客户端
   * 使用提供商配置创建AWS SDK客户端实例
   */
  private createBedrockRuntimeClient(): BedrockRuntimeClient {
    if (this.provider.type !== 'bedrock') {
      throw new Error('[BedrockAPI] 提供商类型必须是bedrock')
    }

    const clientConfig = {
      region: this.provider.region || BedrockAPIClient.DEFAULT_REGION,
      credentials: {
        accessKeyId: this.provider.accessKey || '',
        secretAccessKey: this.provider.secretKey || ''
      }
    }

    return new BedrockRuntimeClient(clientConfig)
  }

  /**
   * 获取模型ID
   * 支持跨区域模型配置
   */
  private getModelId(model: Model): string {
    if (this.provider.type !== 'bedrock') {
      return model.id
    }

    // 如果启用跨区域访问，添加区域前缀
    return this.provider.crossRegion ? `us.${model.id}` : model.id
  }

  /**
   * 获取温度参数
   * 推理模型在使用思考模式时不支持温度设置
   */
  override getTemperature(assistant: Assistant, model: Model): number | undefined {
    if (assistant.settings?.reasoning_effort && isReasoningModel(model)) {
      return undefined
    }
    return assistant.settings?.temperature
  }

  /**
   * 获取TopP参数
   * 推理模型在使用思考模式时不支持TopP设置
   */
  override getTopP(assistant: Assistant, model: Model): number | undefined {
    if (assistant.settings?.reasoning_effort && isReasoningModel(model)) {
      return undefined
    }
    return assistant.settings?.topP
  }

  /**
   * 保存思考块数据
   * 用于递归工具调用时保持思考上下文
   */
  public saveThinkingBlock(thinkingBlock: { thinking: string; signature: string }): void {
    this.lastThinkingBlock = thinkingBlock
  }

  /**
   * 添加到全局累积使用量
   */
  public addToGlobalUsage(usage: TokenUsage): void {
    this.globalAccumulatedUsage.inputTokens += usage.inputTokens || 0
    this.globalAccumulatedUsage.outputTokens += usage.outputTokens || 0
    this.globalAccumulatedUsage.totalTokens += usage.totalTokens || 0
    console.log('[BedrockAPI] Global accumulated usage updated:', this.globalAccumulatedUsage)
  }

  /**
   * 获取全局累积使用量
   */
  public getGlobalUsage(): { inputTokens: number; outputTokens: number; totalTokens: number } {
    return { ...this.globalAccumulatedUsage }
  }

  /**
   * 重置全局累积使用量（新会话开始时）
   */
  public resetGlobalUsage(): void {
    this.globalAccumulatedUsage = {
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0
    }
    console.log('[BedrockAPI] Global usage reset')
  }

  /**
   * 处理工具调用数据 - 公共方法
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
   * 类型判断：是否为Converse命令输出
   */
  private isConverseCommandOutput(output: BedrockSdkRawOutput | string | undefined): output is ConverseCommandOutput {
    return typeof output === 'object' && output !== null && 'output' in output && 'stopReason' in output
  }

  /**
   * 获取推理思考预算配置
   * 用于控制推理模型的思考令牌使用量
   */
  private buildReasoningConfig(assistant: Assistant, model: Model): Record<string, any> | undefined {
    if (!isReasoningModel(model)) {
      return undefined
    }

    const { maxTokens } = getAssistantSettings(assistant)
    const reasoningEffort = assistant?.settings?.reasoning_effort

    // 如果未设置推理努力程度，禁用思考模式
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
   * 基于模型限制、努力程度和最大令牌数计算合适的预算
   */
  private calculateThinkingBudget(model: Model, reasoningEffort: string, maxTokens?: number): number {
    const effortRatio = EFFORT_RATIO[reasoningEffort] || 0.5
    const tokenLimit = findTokenLimit(model.id)

    if (!tokenLimit) {
      return BedrockAPIClient.MIN_THINKING_BUDGET_TOKENS
    }

    const dynamicBudget = (tokenLimit.max - tokenLimit.min) * effortRatio + tokenLimit.min
    const maxAllowedBudget = (maxTokens || DEFAULT_MAX_TOKENS) * effortRatio

    return Math.max(BedrockAPIClient.MIN_THINKING_BUDGET_TOKENS, Math.floor(Math.min(dynamicBudget, maxAllowedBudget)))
  }

  /**
   * 转换载荷消息为Bedrock格式
   * 确保消息格式符合AWS SDK要求
   */
  private transformMessagesForBedrock(messages: BedrockSdkMessageParam[]) {
    return messages.map((message) => ({
      role: message.role as ConversationRole,
      content: message.content as BedrockContentBlock[]
    }))
  }

  /**
   * 转换单个消息为SDK参数格式
   * 处理文本、图像和文件内容
   */
  public async convertMessageToSdkParam(message: Message, model: Model): Promise<BedrockSdkMessageParam> {
    const isVisionCapable = isVisionModel(model)
    const messageContent = await this.getMessageContent(message)
    const contentBlocks: ContentBlock[] = []

    // 添加文本内容
    if (messageContent) {
      contentBlocks.push({ text: messageContent })
    }

    // 处理图像内容（仅限视觉模型）
    await this.processImageContent(message, contentBlocks, isVisionCapable)

    // 处理文件内容
    await this.processFileContent(message, contentBlocks)

    return {
      role: message.role === 'system' ? 'user' : message.role,
      content: contentBlocks
    } as BedrockSdkMessageParam
  }

  /**
   * 处理图像内容
   * 将图像文件转换为Bedrock可识别的格式
   */
  private async processImageContent(
    message: Message,
    contentBlocks: ContentBlock[],
    isVisionCapable: boolean
  ): Promise<void> {
    if (!isVisionCapable) {
      return
    }

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
   * 支持JPEG、PNG等常见格式
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
   * 支持文本和文档类型文件
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
   * 将Cherry Studio的工具定义转换为Bedrock工具规范
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
   */
  convertSdkToolCallToMcp(toolCall: BedrockSdkToolCall, mcpTools: MCPTool[]): MCPTool | undefined {
    return mcpTools.find((tool) => tool.name === toolCall.name)
  }

  /**
   * 转换SDK工具调用为MCP工具响应
   */
  convertSdkToolCallToMcpToolResponse(toolCall: BedrockSdkToolCall, mcpTool: MCPTool): ToolCallResponse {
    // 确保 arguments 是对象格式，不是字符串
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
   * 将工具执行结果转换为Bedrock消息格式
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
   * 支持多轮对话和工具调用的消息构建
   */
  override buildSdkMessages(
    currentMessages: BedrockSdkMessageParam[],
    output: BedrockSdkRawOutput | string | undefined,
    toolResults: BedrockSdkMessageParam[],
    toolCalls?: BedrockSdkToolCall[]
  ): BedrockSdkMessageParam[] {
    const messages = [...currentMessages]

    // 处理Bedrock响应输出
    if (this.isConverseCommandOutput(output)) {
      const assistantContent = output.output?.message?.content
      if (assistantContent) {
        const assistantMessage: BedrockSdkMessageParam = {
          role: 'assistant',
          content: assistantContent
        }
        messages.push(assistantMessage)
      }
    } else {
      // 手动构建助手消息（递归工具调用场景）
      const assistantContent: BedrockContentBlock[] = []
      const hasToolCalls = toolCalls && toolCalls.length > 0

      // 对于工具调用，Bedrock要求先有思考内容
      if (hasToolCalls) {
        const thinkingText =
          this.lastThinkingBlock?.thinking || (typeof output === 'string' && output.trim() ? output : '')

        // 当启用思考模式时，必须使用思考块而不是文本块
        if (thinkingText && thinkingText.trim()) {
          assistantContent.push({
            reasoningContent: {
              reasoningText: {
                text: thinkingText,
                signature: this.lastThinkingBlock?.signature
              } as ReasoningTextBlock
            }
          })
        }

        // 添加工具调用
        for (const toolCall of toolCalls) {
          // 确保 input 是 JSON 对象而不是字符串
          let parsedInput: any
          try {
            parsedInput = typeof toolCall.input === 'string' ? JSON.parse(toolCall.input) : toolCall.input
          } catch (error) {
            console.error('[BedrockAPI] Failed to parse tool input:', toolCall.input, error)
            parsedInput = {} // 使用空对象作为备选
          }

          assistantContent.push({
            toolUse: {
              toolUseId: toolCall.toolUseId,
              name: toolCall.name,
              input: parsedInput || {}
            }
          })
        }
      } else if (typeof output === 'string' && output.trim()) {
        // 纯文本响应
        assistantContent.push({ text: output })
      }

      // 只在有内容时添加助手消息
      if (assistantContent.length > 0) {
        messages.push({
          role: 'assistant',
          content: assistantContent
        })
      }
    }

    // 添加工具执行结果
    if (toolResults && toolResults.length > 0) {
      messages.push(...toolResults)
    }

    return messages
  }

  /**
   * 估算消息令牌数
   * 用于令牌使用量预估和限制
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
   */
  extractMessagesFromSdkPayload(sdkPayload: BedrockSdkParams): BedrockSdkMessageParam[] {
    return sdkPayload.messages || []
  }

  /**
   * 获取请求转换器
   * 将Cherry Studio的请求格式转换为Bedrock SDK格式
   */
  getRequestTransformer(): RequestTransformer<BedrockSdkParams, BedrockSdkMessageParam> {
    return {
      transform: async (coreRequest, assistant, model, isRecursiveCall, recursiveSdkMessages) => {
        const { messages, mcpTools, maxTokens, streamOutput } = coreRequest

        // 配置工具设置
        this.setupToolsConfig({ mcpTools, model, enableToolUse: true })
        const tools = this.useSystemPromptForTools ? [] : mcpTools ? this.convertMcpToolsToSdkTools(mcpTools) : []

        // 构建系统提示
        let systemContent = assistant.prompt || ''
        if (this.useSystemPromptForTools && mcpTools) {
          systemContent = await buildSystemPrompt(systemContent, mcpTools, assistant)
        }

        // 处理用户消息
        const userMessages = await this.processUserMessages(messages, model)
        const requestMessages = isRecursiveCall && recursiveSdkMessages?.length ? recursiveSdkMessages : userMessages

        // 构建推理配置
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
   * 将字符串或消息数组转换为SDK参数格式
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
   * 设置模型推理参数
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
   * 使用新的事件处理架构处理流式响应
   */
  getResponseChunkTransformer(): ResponseChunkTransformer<BedrockSdkRawChunk> {
    return () => {
      const streamState = new BedrockStreamState()
      const streamProcessor = new BedrockStreamProcessor(this)

      // 不重置全局累积使用量，让它在整个会话中保持
      // streamState.resetAccumulatedUsage() // 删除这行

      return {
        transform: (chunk: BedrockSdkRawChunk, controller: TransformStreamDefaultController<GenericChunk>) => {
          console.log('chunk', chunk)
          streamProcessor.processEvent(chunk, streamState, controller)
        }
      }
    }
  }
}
