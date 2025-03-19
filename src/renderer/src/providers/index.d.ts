import type { GroundingMetadata } from '@google/generative-ai'
import type { Assistant, MCPTool, MCPToolResponse, Message, Metrics } from '@renderer/types'

interface CompletionUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

interface ChunkCallbackData {
  text?: string
  reasoning_content?: string
  usage?: CompletionUsage
  metrics?: Metrics
  search?: GroundingMetadata
  citations?: string[]
  mcpToolResponse?: MCPToolResponse[]
}

interface CompletionsParams {
  messages: Message[]
  assistant: Assistant
  onChunk: ({ text, reasoning_content, usage, metrics, search, citations, mcpToolResponse }: ChunkCallbackData) => void
  onFilterMessages: (messages: Message[]) => void
  mcpTools?: MCPTool[]
}
