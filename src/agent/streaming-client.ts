import { AIMessage } from "@langchain/core/messages";

/**
 * xAI Grok-2 Vision API client with streaming support
 */
export class GrokStreamingClient {
  private apiKey: string;
  private baseURL: string;

  constructor(apiKey: string, baseURL: string = "https://api.x.ai/v1") {
    this.apiKey = apiKey;
    this.baseURL = baseURL;
  }

  async *streamChatCompletion(messages: any[], tools: any[] = []) {
    const data = {
      model: "grok-2-vision-1212",
      messages,
      tools: tools.length > 0 ? tools : undefined,
      tool_choice: tools.length > 0 ? "auto" : undefined,
      stream: true,
      temperature: 0.1,
      max_tokens: 4096,
    };

    const response = await fetch(`${this.baseURL}/chat/completions`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`Grok API error: ${response.status} - ${response.statusText}`);
    }

    if (!response.body) {
      throw new Error("No response body available for streaming");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let buffer = "";
    let accumulatedContent = "";
    const accumulatedToolCalls: any[] = [];
    const responseMetadata: any = {};

    try {
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        const lines = buffer.split('\n');
        buffer = lines.pop() || "";
        
        for (const line of lines) {
          const trimmedLine = line.trim();
          if (trimmedLine.startsWith('data: ')) {
            const dataStr = trimmedLine.slice(6);
            
            if (dataStr === '[DONE]') {
              // Final message
              yield new AIMessage({
                content: accumulatedContent,
                tool_calls: accumulatedToolCalls,
                response_metadata: responseMetadata,
              });
              return;
            }
            
            try {
              const jsonData = JSON.parse(dataStr);
              
              if (jsonData.choices && jsonData.choices[0]) {
                const delta = jsonData.choices[0].delta;
                
                // Update metadata if available
                if (jsonData.id) {
                  responseMetadata.id = jsonData.id;
                }
                if (jsonData.model) {
                  responseMetadata.model = jsonData.model;
                }
                if (jsonData.usage) {
                  responseMetadata.usage = jsonData.usage;
                }
                
                // Handle content streaming
                if (delta.content) {
                  accumulatedContent += delta.content;
                  
                  yield new AIMessage({
                    content: accumulatedContent,
                    tool_calls: accumulatedToolCalls,
                    response_metadata: responseMetadata,
                  });
                }
                
                // Handle tool calls
                if (delta.tool_calls) {
                  for (const toolCallDelta of delta.tool_calls) {
                    const index = toolCallDelta.index || 0;
                    
                    // Initialize tool call if needed
                    while (accumulatedToolCalls.length <= index) {
                      accumulatedToolCalls.push({
                        id: "",
                        name: "",
                        args: {},
                      });
                    }
                    
                    const toolCall = accumulatedToolCalls[index];
                    
                    if (toolCallDelta.id) {
                      toolCall.id = toolCallDelta.id;
                    }
                    
                    if (toolCallDelta.function) {
                      if (toolCallDelta.function.name) {
                        toolCall.name = toolCallDelta.function.name;
                      }
                      
                      if (toolCallDelta.function.arguments) {
                        toolCall.args = {
                          ...toolCall.args,
                          ...JSON.parse(toolCallDelta.function.arguments),
                        };
                      }
                    }
                  }
                  
                  yield new AIMessage({
                    content: accumulatedContent,
                    tool_calls: accumulatedToolCalls,
                    response_metadata: responseMetadata,
                  });
                }
              }
            } catch (error) {
              console.warn("Failed to parse streaming data:", dataStr, error);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
    
    // Final message if not already yielded
    yield new AIMessage({
      content: accumulatedContent,
      tool_calls: accumulatedToolCalls,
      response_metadata: responseMetadata,
    });
  }
}

/**
 * Enhanced streaming model call implementation
 */
export async function* callModelStreamingCustom(
  state: any,
  config: any
): AsyncGenerator<{ messages: AIMessage }, void, unknown> {
  const configuration = { 
    environment: "ubuntu",
    zdrEnabled: false,
    prompt: undefined,
    ...config?.configurable 
  };

  // Get xAI API key from environment
  const xaiApiKey = process.env.XAI_API_KEY;
  if (!xaiApiKey) {
    throw new Error("XAI_API_KEY environment variable is required");
  }

  const grokClient = new GrokStreamingClient(xaiApiKey, process.env.XAI_BASE_URL);

  // Convert messages to Grok format (simplified for now)
  const messages = state.messages.map((msg: any) => {
    if (msg.getType() === "human") {
      return { role: "user", content: msg.content };
    } else if (msg.getType() === "ai") {
      return { role: "assistant", content: msg.content };
    } else if (msg.getType() === "system") {
      return { role: "system", content: msg.content };
    }
    return { role: "user", content: msg.content || "" };
  });

  // Add system prompt if provided
  if (configuration.prompt) {
    messages.unshift({ 
      role: "system", 
      content: typeof configuration.prompt === "string" 
        ? configuration.prompt 
        : configuration.prompt.content 
    });
  }

  // Create computer use tool
  const tools = [{
    type: "function",
    function: {
      name: "computer_use",
      description: "Use a computer to take actions like clicking, typing, and taking screenshots",
      parameters: {
        type: "object",
        properties: {
          action: {
            type: "object",
            description: "The computer action to perform",
            properties: {
              type: {
                type: "string",
                enum: ["click", "double_click", "drag", "type", "key", "screenshot", "scroll", "wait"],
                description: "The type of action to perform",
              },
              x: { type: "number", description: "X coordinate for click/drag actions" },
              y: { type: "number", description: "Y coordinate for click/drag actions" },
              button: { type: "string", enum: ["left", "right", "middle"] },
              text: { type: "string", description: "Text to type" },
              key: { type: "string", description: "Key to press" },
              scroll_direction: { type: "string", enum: ["up", "down", "left", "right"] },
              scroll_amount: { type: "number" },
              duration: { type: "number" },
            },
            required: ["type"],
          },
        },
        required: ["action"],
      },
    },
  }];

  // Stream the response
  for await (const response of grokClient.streamChatCompletion(messages, tools)) {
    yield { messages: response };
  }
}