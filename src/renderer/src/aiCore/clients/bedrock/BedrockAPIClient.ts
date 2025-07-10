import {
  BedrockRuntimeClient,
  ContentBlock as BedrockContentBlock,
  ContentBlock,
  ConversationRole,
  ConverseCommand,
  ConverseStreamCommand
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
import { ChunkType, ThinkingDeltaChunk } from '@renderer/types/chunk'
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
 * Amazon Bedrock API 客户端
 * 负责处理与 Amazon Bedrock 服务的所有交互，包括消息转换、工具调用和流响应处理
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
  private static readonly DEFAULT_REGION = 'us-east-1'
  private static readonly MIN_BUDGET_TOKENS = 1024

  private client?: BedrockRuntimeClient

  constructor(provider: Provider) {
    super(provider)
  }

  /**
   * 生成图像 - Bedrock SDK 暂未提供此功能
   */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  override async generateImage(_generateImageParams: GenerateImageParams): Promise<string[]> {
    return []
  }

  /**
   * 获取嵌入向量维度 - Bedrock SDK 暂未提供此功能
   */
  override async getEmbeddingDimensions(): Promise<number> {
    throw new Error('Bedrock SDK 暂不支持 getEmbeddingDimensions 方法')
  }

  /**
   * 列出可用模型
   */
  override async listModels(): Promise<SdkModel[]> {
    return []
  }

  /**
   * 创建聊天完成请求
   */
  override async createCompletions(payload: BedrockSdkParams): Promise<BedrockSdkRawOutput> {
    const client = await this.getSdkInstance()
    const messages = this.convertPayloadMessages(payload.messages)

    const commandParams = {
      modelId: payload.modelId,
      messages,
      system: payload.system,
      inferenceConfig: payload.inferenceConfig,
      toolConfig: payload.toolConfig,
      additionalModelRequestFields: payload.additionalModelRequestFields
    }

    if (payload.stream) {
      const command = new ConverseStreamCommand(commandParams)
      const response = await client.send(command)
      return response.stream as BedrockSdkRawOutput
    } else {
      const command = new ConverseCommand(commandParams)
      const response = await client.send(command)
      return response as BedrockSdkRawOutput
    }
  }

  /**
   * 获取 Bedrock SDK 实例
   */
  async getSdkInstance(): Promise<BedrockSdkInstance> {
    if (!this.client) {
      this.client = this.createBedrockClient()
    }
    return this.client as BedrockSdkInstance
  }

  /**
   * 创建 Bedrock 客户端实例
   */
  private createBedrockClient(): BedrockRuntimeClient {
    if (this.provider.type !== 'bedrock') {
      throw new Error('提供商不是 Bedrock 提供商')
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
   * 获取模型 ID（支持跨区域模型）
   */
  private getModelId(model: Model): string {
    console.log('this.provider', this.provider)
    if (this.provider.type !== 'bedrock') {
      return model.id
    }
    return this.provider.crossRegion ? `us.${model.id}` : model.id
  }

  /**
   * 获取温度参数（推理模型时返回 undefined）
   */
  override getTemperature(assistant: Assistant, model: Model): number | undefined {
    if (assistant.settings?.reasoning_effort && isReasoningModel(model)) {
      return undefined
    }
    return assistant.settings?.temperature
  }

  /**
   * 获取 TopP 参数（推理模型时返回 undefined）
   */
  override getTopP(assistant: Assistant, model: Model): number | undefined {
    if (assistant.settings?.reasoning_effort && isReasoningModel(model)) {
      return undefined
    }
    return assistant.settings?.topP
  }

  /**
   * 获取推理思考预算配置
   */
  private getReasoningBudgetConfig(assistant: Assistant, model: Model): Record<string, any> | undefined {
    if (!isReasoningModel(model)) {
      return undefined
    }

    const { maxTokens } = getAssistantSettings(assistant)
    const reasoningEffort = assistant?.settings?.reasoning_effort

    if (reasoningEffort === undefined) {
      return { thinking: { type: 'disabled' } }
    }

    const budgetTokens = this.calculateBudgetTokens(model, reasoningEffort, maxTokens)

    return {
      thinking: {
        type: 'enabled',
        budget_tokens: budgetTokens
      }
    }
  }

  /**
   * 计算思考预算令牌数
   */
  private calculateBudgetTokens(model: Model, reasoningEffort: string, maxTokens?: number): number {
    const effortRatio = EFFORT_RATIO[reasoningEffort]
    const tokenLimit = findTokenLimit(model.id)

    if (!tokenLimit) {
      return BedrockAPIClient.MIN_BUDGET_TOKENS
    }

    const dynamicTokens = (tokenLimit.max - tokenLimit.min) * effortRatio + tokenLimit.min
    const maxAllowedTokens = (maxTokens || DEFAULT_MAX_TOKENS) * effortRatio

    return Math.max(BedrockAPIClient.MIN_BUDGET_TOKENS, Math.floor(Math.min(dynamicTokens, maxAllowedTokens)))
  }

  /**
   * 转换载荷中的消息格式
   */
  private convertPayloadMessages(messages: BedrockSdkMessageParam[]) {
    return messages.map((msg) => ({
      role: msg.role as ConversationRole,
      content: msg.content as BedrockContentBlock[]
    }))
  }

  /**
   * 将消息转换为 SDK 参数格式
   */
  public async convertMessageToSdkParam(message: Message, model: Model): Promise<BedrockSdkMessageParam> {
    const isVision = isVisionModel(model)
    const content = await this.getMessageContent(message)
    const contentBlocks: ContentBlock[] = []

    // 添加文本内容
    if (content) {
      contentBlocks.push({ text: content })
    }

    // 处理图像内容
    await this.processImageBlocks(message, contentBlocks, isVision)

    // 处理文件内容
    await this.processFileBlocks(message, contentBlocks)

    return {
      role: message.role === 'system' ? 'user' : message.role,
      content: contentBlocks
    } as BedrockSdkMessageParam
  }

  /**
   * 处理图像块
   */
  private async processImageBlocks(message: Message, contentBlocks: ContentBlock[], isVision: boolean) {
    const imageBlocks = findImageBlocks(message)

    for (const imageBlock of imageBlocks) {
      if (isVision && imageBlock.file) {
        const imageContent = await this.convertImageToContentBlock(imageBlock.file)
        if (imageContent) {
          contentBlocks.push(imageContent)
        }
      }
    }
  }

  /**
   * 转换图像文件为内容块
   */
  private async convertImageToContentBlock(file: any): Promise<ContentBlock | null> {
    try {
      const image = await window.api.file.base64Image(file.id + file.ext)
      const base64Data = image.data.split(',')[1]
      const format = image.data.includes('jpeg') ? 'jpeg' : 'png'

      return {
        image: {
          format: format as 'jpeg' | 'png' | 'gif' | 'webp',
          source: { bytes: new Uint8Array(Buffer.from(base64Data, 'base64')) }
        }
      }
    } catch (error) {
      console.error('转换图像文件失败:', error)
      return null
    }
  }

  /**
   * 处理文件块
   */
  private async processFileBlocks(message: Message, contentBlocks: ContentBlock[]) {
    const fileBlocks = findFileBlocks(message)

    for (const fileBlock of fileBlocks) {
      const file = fileBlock.file
      if (file && [FileTypes.TEXT, FileTypes.DOCUMENT].includes(file.type)) {
        try {
          const fileContent = await window.api.file.read(file.id + file.ext)
          contentBlocks.push({ text: `${file.origin_name}\n${fileContent.trim()}` })
        } catch (error) {
          console.error('读取文件失败:', error)
        }
      }
    }
  }

  /**
   * 将 MCP 工具转换为 SDK 工具
   */
  convertMcpToolsToSdkTools(mcpTools: MCPTool[]): BedrockSdkTool[] {
    return mcpTools.map((tool) => ({
      toolSpec: {
        name: tool.name,
        description: tool.description,
        inputSchema: {
          json: tool.inputSchema
        }
      }
    }))
  }

  /**
   * 根据工具调用查找对应的 MCP 工具
   */
  convertSdkToolCallToMcp(toolCall: BedrockSdkToolCall, mcpTools: MCPTool[]): MCPTool | undefined {
    return mcpTools.find((tool) => tool.name === toolCall.name)
  }

  /**
   * 将 SDK 工具调用转换为 MCP 工具响应
   */
  convertSdkToolCallToMcpToolResponse(toolCall: BedrockSdkToolCall, mcpTool: MCPTool): ToolCallResponse {
    return {
      id: toolCall.toolUseId,
      toolCallId: toolCall.toolUseId,
      tool: mcpTool,
      arguments: toolCall.input,
      status: 'pending'
    } as ToolCallResponse
  }

  /**
   * 将 MCP 工具响应转换为 SDK 消息参数
   */
  convertMcpToolResponseToSdkMessageParam(
    mcpToolResponse: MCPToolResponse,
    resp: MCPCallToolResponse
  ): BedrockSdkMessageParam | undefined {
    const toolUseId = this.extractToolUseId(mcpToolResponse)
    if (!toolUseId) {
      return undefined
    }

    const resultText = this.extractResultText(resp)

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
   * 提取工具使用 ID
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
   * 提取结果文本
   */
  private extractResultText(resp: MCPCallToolResponse): string {
    if (Array.isArray(resp.content) && resp.content.length > 0 && resp.content[0].text) {
      return resp.content.map((c) => c.text || '').join('\n')
    }
    if (typeof resp.content === 'object') {
      return JSON.stringify(resp.content)
    }
    return String(resp.content)
  }

  /**
   * 构建 SDK 消息列表
   */
  override buildSdkMessages(
    currentReqMessages: BedrockSdkMessageParam[],
    output: BedrockSdkRawOutput | string | undefined,
    toolResults: BedrockSdkMessageParam[],
    toolCalls?: BedrockSdkToolCall[]
  ): BedrockSdkMessageParam[] {
    const hasTextOutput = typeof output === 'string' && output.trim().length > 0
    const hasToolCalls = toolCalls && toolCalls.length > 0

    const assistantMessage = this.createAssistantMessage(output, toolCalls, hasTextOutput, hasToolCalls || false)
    let result = [...currentReqMessages]

    if (assistantMessage.content!.length > 0) {
      result.push(assistantMessage)
    }

    if (toolResults && toolResults.length > 0) {
      result = [...result, ...toolResults]
    }

    return result
  }

  /**
   * 创建助手消息
   */
  private createAssistantMessage(
    output: BedrockSdkRawOutput | string | undefined,
    toolCalls: BedrockSdkToolCall[] | undefined,
    hasTextOutput: boolean,
    hasToolCalls: boolean
  ): BedrockSdkMessageParam {
    const assistantMessage: BedrockSdkMessageParam = {
      role: 'assistant',
      content: []
    }

    if (hasTextOutput) {
      assistantMessage.content!.push({ text: output as string })
    }

    if (hasToolCalls && toolCalls) {
      for (const tool of toolCalls) {
        const contentBlock = {
          toolUse: {
            toolUseId: tool.toolUseId,
            name: tool.name,
            input: tool.input
          }
        } as BedrockContentBlock
        assistantMessage.content!.push(contentBlock)
      }
    }

    return assistantMessage
  }

  /**
   * 估算消息令牌数
   */
  override estimateMessageTokens(message: BedrockSdkMessageParam): number {
    let sum = 0
    if (message.content) {
      for (const block of message.content) {
        if ('text' in block && block.text) {
          sum += estimateTextTokens(block.text)
        }
      }
    }
    return sum
  }

  /**
   * 从 SDK 载荷中提取消息
   */
  extractMessagesFromSdkPayload(sdkPayload: BedrockSdkParams): BedrockSdkMessageParam[] {
    return sdkPayload.messages || []
  }

  /**
   * 获取请求转换器
   */
  getRequestTransformer(): RequestTransformer<BedrockSdkParams, BedrockSdkMessageParam> {
    return {
      transform: async (coreRequest, assistant, model, isRecursiveCall, recursiveSdkMessages) => {
        const { messages, mcpTools, maxTokens, streamOutput } = coreRequest

        // 配置工具
        this.setupToolsConfig({ mcpTools, model, enableToolUse: true })
        const tools = this.useSystemPromptForTools ? [] : mcpTools ? this.convertMcpToolsToSdkTools(mcpTools) : []

        // 构建系统消息
        let systemContent = assistant.prompt || ''
        if (this.useSystemPromptForTools) {
          systemContent = await buildSystemPrompt(systemContent, mcpTools, assistant)
        }

        // 处理用户消息
        const userMessages = await this.processUserMessages(messages, model)
        const reqMessages = isRecursiveCall && recursiveSdkMessages?.length ? recursiveSdkMessages : userMessages

        // 构建推理配置
        const inferenceConfig = this.buildInferenceConfig(assistant, model, maxTokens)
        const reasoningConfig = this.getReasoningBudgetConfig(assistant, model)

        const sdkParams: BedrockSdkParams = {
          modelId: this.getModelId(model),
          messages: reqMessages,
          system: systemContent ? [{ text: systemContent }] : undefined,
          inferenceConfig,
          toolConfig: tools.length > 0 ? { tools } : undefined,
          additionalModelRequestFields: reasoningConfig,
          stream: streamOutput
        }

        const timeout = this.getTimeout(model)
        return { payload: sdkParams, messages: reqMessages, metadata: { timeout } }
      }
    }
  }

  /**
   * 处理用户消息
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
   */
  private buildInferenceConfig(assistant: Assistant, model: Model, maxTokens?: number) {
    return {
      maxTokens: maxTokens || DEFAULT_MAX_TOKENS,
      temperature: this.getTemperature(assistant, model),
      topP: this.getTopP(assistant, model)
    }
  }

  /**
   * 获取响应块转换器
   */
  getResponseChunkTransformer(): ResponseChunkTransformer<BedrockSdkRawChunk> {
    return () => {
      const toolCalls: BedrockSdkToolCall[] = []
      const usageContainer = { current: null as any }

      return {
        transform: (chunk: BedrockSdkRawChunk, controller: TransformStreamDefaultController<GenericChunk>) => {
          if (!chunk) {
            console.warn('从 Bedrock API 接收到空块')
            return
          }

          if ('output' in chunk) {
            this.handleNonStreamingChunk(chunk, controller, toolCalls, usageContainer)
          } else {
            this.handleStreamingChunk(chunk, controller, toolCalls, usageContainer)
          }
        }
      }
    }
  }

  /**
   * 处理非流式响应块
   */
  private handleNonStreamingChunk(
    chunk: any,
    controller: TransformStreamDefaultController<GenericChunk>,
    toolCalls: BedrockSdkToolCall[],
    usageContainer: { current: any }
  ) {
    if (chunk.usage) {
      usageContainer.current = chunk.usage
    }

    chunk.output?.message?.content?.forEach((item: any) => {
      if (item.text) {
        controller.enqueue({ type: ChunkType.TEXT_DELTA, text: item.text })
      }
      if (item.toolUse) {
        toolCalls.push({
          toolUseId: item.toolUse.toolUseId || '',
          name: item.toolUse.name || '',
          input: item.toolUse.input || {},
          type: 'tool_use'
        })
      }
    })

    this.handleResponseCompletion(chunk.stopReason, controller, toolCalls, usageContainer.current)
  }

  /**
   * 处理流式响应块
   */
  private handleStreamingChunk(
    chunk: any,
    controller: TransformStreamDefaultController<GenericChunk>,
    toolCalls: BedrockSdkToolCall[],
    usageContainer: { current: any }
  ) {
    const streamChunk = chunk as any

    // 处理工具使用开始
    if (streamChunk.contentBlockStart?.start?.toolUse) {
      this.handleToolUseStart(streamChunk, toolCalls)
    }

    // 处理内容增量
    if (streamChunk.contentBlockDelta?.delta) {
      this.handleContentDelta(streamChunk, controller, toolCalls)
    }

    // 处理使用情况元数据
    if (streamChunk.metadata?.usage) {
      usageContainer.current = streamChunk.metadata
      this.sendUsageMetrics(controller, usageContainer.current)
    }

    // 处理消息停止
    if (streamChunk.messageStop?.stopReason) {
      this.handleResponseCompletion(streamChunk.messageStop.stopReason, controller, toolCalls, usageContainer.current)
    }
  }

  /**
   * 处理工具使用开始
   */
  private handleToolUseStart(streamChunk: any, toolCalls: BedrockSdkToolCall[]) {
    try {
      const { toolUseId, name } = streamChunk.contentBlockStart.start.toolUse
      toolCalls.push({
        toolUseId: toolUseId || '',
        name: name || '',
        input: '',
        type: 'tool_use'
      })
    } catch (error) {
      console.warn('处理工具使用开始时出错:', error)
    }
  }

  /**
   * 处理内容增量
   */
  private handleContentDelta(
    streamChunk: any,
    controller: TransformStreamDefaultController<GenericChunk>,
    toolCalls: BedrockSdkToolCall[]
  ) {
    const delta = streamChunk.contentBlockDelta.delta

    if (delta.text) {
      this.handleTextDelta(delta.text, controller)
    }

    if (delta.toolUse?.input && toolCalls.length > 0) {
      this.handleToolInputDelta(delta.toolUse.input, toolCalls)
    }

    if (delta.reasoningContent?.text) {
      this.handleReasoningDelta(delta.reasoningContent.text, controller)
    }
  }

  /**
   * 处理文本增量
   */
  private handleTextDelta(text: string, controller: TransformStreamDefaultController<GenericChunk>) {
    controller.enqueue({ type: ChunkType.TEXT_DELTA, text })
  }

  /**
   * 处理工具输入增量
   */
  private handleToolInputDelta(input: string, toolCalls: BedrockSdkToolCall[]) {
    try {
      toolCalls[toolCalls.length - 1].input += input
    } catch (error) {
      console.warn('处理工具输入时出错:', error)
    }
  }

  /**
   * 处理推理增量
   */
  private handleReasoningDelta(text: string, controller: TransformStreamDefaultController<GenericChunk>) {
    controller.enqueue({
      type: ChunkType.THINKING_DELTA,
      text
    } as ThinkingDeltaChunk)
  }

  /**
   * 发送使用情况指标
   */
  private sendUsageMetrics(controller: TransformStreamDefaultController<GenericChunk>, usage: any) {
    controller.enqueue({
      type: ChunkType.LLM_RESPONSE_COMPLETE,
      response: {
        usage: {
          prompt_tokens: usage?.usage?.inputTokens || 0,
          completion_tokens: usage?.usage?.outputTokens || 0,
          total_tokens: usage?.usage?.totalTokens || 0
        }
      }
    })
  }

  /**
   * 处理响应完成
   */
  private handleResponseCompletion(
    stopReason: string,
    controller: TransformStreamDefaultController<GenericChunk>,
    toolCalls: BedrockSdkToolCall[],
    usage: any
  ) {
    if (stopReason === 'tool_use' && toolCalls.length > 0) {
      const completedToolCalls = this.parseToolCalls(toolCalls)
      controller.enqueue({ type: ChunkType.MCP_TOOL_CREATED, tool_calls: completedToolCalls })
    } else {
      // 发送响应完成信号，让 TextChunkMiddleware 处理文本完成
      this.sendUsageMetrics(controller, usage)
    }
  }

  /**
   * 解析工具调用
   */
  private parseToolCalls(toolCalls: BedrockSdkToolCall[]): BedrockSdkToolCall[] {
    return toolCalls.map((tc) => {
      try {
        const parsedInput = typeof tc.input === 'string' && tc.input ? JSON.parse(tc.input) : tc.input
        return { ...tc, input: parsedInput || {} }
      } catch (e) {
        console.error('解析工具调用输入 JSON 时出错:', tc.input, e)
        return { ...tc, input: {} }
      }
    })
  }
}
