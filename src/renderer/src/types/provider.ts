import { Provider } from './index'

/**
 * Typeguard to check if a provider is a Bedrock provider
 */
export function isBedrock(provider: Provider): provider is Provider & { 
  type: 'bedrock'
  region?: string
  accessKey?: string
  secretKey?: string
  crossRegion?: boolean
} {
  return provider.type === 'bedrock'
}