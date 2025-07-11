import Anthropic from '@anthropic-ai/sdk'
import {
  Message,
  MessageCreateParams,
  MessageParam,
  RawMessageStreamEvent,
  ToolUnion,
  ToolUseBlock
} from '@anthropic-ai/sdk/resources'
import { MessageStream } from '@anthropic-ai/sdk/resources/messages/messages'
import {
  BedrockRuntimeClient,
  ContentBlock,
  ContentBlockDeltaEvent,
  ContentBlockStartEvent,
  ContentBlockStopEvent,
  ConverseResponse,
  ConverseStreamMetadataEvent,
  ConverseStreamOutput,
  InferenceConfiguration,
  MessageStartEvent,
  MessageStopEvent,
  SystemContentBlock,
  Tool as BedrockTool,
  ToolConfiguration
} from '@aws-sdk/client-bedrock-runtime'
import {
  Content,
  CreateChatParameters,
  FunctionCall,
  GenerateContentResponse,
  GoogleGenAI,
  Model as GeminiModel,
  SendMessageParameters,
  Tool as GeminiTool
} from '@google/genai'
import OpenAI, { AzureOpenAI } from 'openai'
import { Stream } from 'openai/streaming'

import { EndpointType } from './index'

export type SdkInstance = OpenAI | AzureOpenAI | Anthropic | GoogleGenAI | BedrockRuntimeClient
export type BedrockSdkInstance = BedrockRuntimeClient
export type SdkParams =
  | OpenAISdkParams
  | OpenAIResponseSdkParams
  | AnthropicSdkParams
  | GeminiSdkParams
  | BedrockSdkParams
export type SdkRawChunk =
  | OpenAISdkRawChunk
  | OpenAIResponseSdkRawChunk
  | AnthropicSdkRawChunk
  | GeminiSdkRawChunk
  | BedrockSdkRawChunk
export type SdkRawOutput =
  | OpenAISdkRawOutput
  | OpenAIResponseSdkRawOutput
  | AnthropicSdkRawOutput
  | GeminiSdkRawOutput
  | BedrockSdkRawOutput
export type SdkMessageParam =
  | OpenAISdkMessageParam
  | OpenAIResponseSdkMessageParam
  | AnthropicSdkMessageParam
  | GeminiSdkMessageParam
  | BedrockSdkMessageParam
export type SdkToolCall =
  | OpenAI.Chat.Completions.ChatCompletionMessageToolCall
  | ToolUseBlock
  | FunctionCall
  | OpenAIResponseSdkToolCall
  | BedrockSdkToolCall
export type SdkTool =
  | OpenAI.Chat.Completions.ChatCompletionTool
  | ToolUnion
  | GeminiTool
  | OpenAIResponseSdkTool
  | BedrockSdkTool
export type SdkModel = OpenAI.Models.Model | Anthropic.ModelInfo | GeminiModel | NewApiModel

export type RequestOptions = Anthropic.RequestOptions | OpenAI.RequestOptions | GeminiOptions

/**
 * OpenAI
 */

type OpenAIParamsWithoutReasoningEffort = Omit<OpenAI.Chat.Completions.ChatCompletionCreateParams, 'reasoning_effort'>

export type ReasoningEffortOptionalParams = {
  thinking?: { type: 'disabled' | 'enabled' | 'auto'; budget_tokens?: number }
  reasoning?: { max_tokens?: number; exclude?: boolean; effort?: string; enabled?: boolean } | OpenAI.Reasoning
  reasoning_effort?: OpenAI.Chat.Completions.ChatCompletionCreateParams['reasoning_effort'] | 'none' | 'auto'
  enable_thinking?: boolean
  thinking_budget?: number
  incremental_output?: boolean
  enable_reasoning?: boolean
  extra_body?: Record<string, any>
  // Add any other potential reasoning-related keys here if they exist
}

export type OpenAISdkParams = OpenAIParamsWithoutReasoningEffort & ReasoningEffortOptionalParams
export type OpenAISdkRawChunk =
  | OpenAI.Chat.Completions.ChatCompletionChunk
  | ({
      _request_id?: string | null | undefined
    } & OpenAI.ChatCompletion)

export type OpenAISdkRawOutput = Stream<OpenAI.Chat.Completions.ChatCompletionChunk> | OpenAI.ChatCompletion
export type OpenAISdkRawContentSource =
  | OpenAI.Chat.Completions.ChatCompletionChunk.Choice.Delta
  | OpenAI.Chat.Completions.ChatCompletionMessage

export type OpenAISdkMessageParam = OpenAI.Chat.Completions.ChatCompletionMessageParam

/**
 * OpenAI Response
 */

export type OpenAIResponseSdkParams = OpenAI.Responses.ResponseCreateParams
export type OpenAIResponseSdkRawOutput = Stream<OpenAI.Responses.ResponseStreamEvent> | OpenAI.Responses.Response
export type OpenAIResponseSdkRawChunk = OpenAI.Responses.ResponseStreamEvent | OpenAI.Responses.Response
export type OpenAIResponseSdkMessageParam = OpenAI.Responses.ResponseInputItem
export type OpenAIResponseSdkToolCall = OpenAI.Responses.ResponseFunctionToolCall
export type OpenAIResponseSdkTool = OpenAI.Responses.Tool

/**
 * Anthropic
 */

export type AnthropicSdkParams = MessageCreateParams
export type AnthropicSdkRawOutput = MessageStream | Message
export type AnthropicSdkRawChunk = RawMessageStreamEvent | Message
export type AnthropicSdkMessageParam = MessageParam

/**
 * Gemini
 */

export type GeminiSdkParams = SendMessageParameters & CreateChatParameters
export type GeminiSdkRawOutput = AsyncGenerator<GenerateContentResponse> | GenerateContentResponse
export type GeminiSdkRawChunk = GenerateContentResponse
export type GeminiSdkMessageParam = Content
export type GeminiSdkToolCall = FunctionCall

export type GeminiOptions = {
  streamOutput: boolean
  signal?: AbortSignal
  timeout?: number
}

/**
 * New API
 */
export interface NewApiModel extends OpenAI.Models.Model {
  supported_endpoint_types?: EndpointType[]
}

/**
 * Bedrock - 基于AWS SDK原生类型的优雅定义
 */

export type BedrockSdkParams = {
  modelId: string
  messages: BedrockSdkMessageParam[]
  system?: SystemContentBlock[]
  inferenceConfig?: InferenceConfiguration
  toolConfig?: ToolConfiguration
  additionalModelRequestFields?: Record<string, any>
  stream?: boolean
}

// 使用AWS SDK的联合类型，支持流式和非流式响应
export type BedrockSdkRawOutput = AsyncIterable<ConverseStreamOutput> | ConverseResponse

// 流式响应的各种事件类型
export type BedrockSdkRawChunk =
  | MessageStartEvent
  | ContentBlockStartEvent
  | ContentBlockDeltaEvent
  | ContentBlockStopEvent
  | MessageStopEvent
  | ConverseResponse
  | ConverseStreamMetadataEvent

export type BedrockSdkMessageParam = {
  role: 'user' | 'assistant' | 'system'
  content: ContentBlock[]
}

// 直接使用AWS SDK的ContentBlock，更加完整和准确
export type BedrockSdkContentBlock = ContentBlock

export type BedrockSdkToolCall = {
  toolUseId: string
  name: string
  input: any
  type: 'tool_use'
}

// 使用AWS SDK的原生Tool类型
export type BedrockSdkTool = BedrockTool

export type BedrockOptions = {
  streamOutput: boolean
  signal?: AbortSignal
  timeout?: number
}
