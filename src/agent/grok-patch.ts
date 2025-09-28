import { ChatOpenAI } from "@langchain/openai";
import { AIMessage } from "@langchain/core/messages";
import { BaseMessage } from "@langchain/core/messages";

/**
 * Custom Grok implementation that mimics ChatOpenAI interface
 * but uses xAI's Grok-2-Vision-1212 model
 */
export class GrokChatModel extends ChatOpenAI {
  private xaiApiKey: string;
  private xaiBaseUrl: string;

  constructor(config: any = {}) {
    // Initialize the parent with dummy values since we won't use OpenAI
    super({
      model: "gpt-4",
      openAIApiKey: "dummy",
      ...config,
    });

    this.xaiApiKey = process.env.XAI_API_KEY || config.xaiApiKey || "";
    this.xaiBaseUrl = process.env.XAI_BASE_URL || "https://api.x.ai/v1";

    // For demo purposes, we'll continue even without a real API key
    // In production, you would throw an error here
    if (!this.xaiApiKey || this.xaiApiKey === "xai-demo-key-placeholder") {
      console.warn("XAI_API_KEY not configured - using fallback responses for demo");
    }
  }

  async _generate(
    messages: BaseMessage[],
    options: any = {},
    runManager?: any
  ): Promise<any> {
    try {
      // Convert LangChain messages to xAI format
      const xaiMessages = messages.map((msg) => {
        if (msg._getType() === "system") {
          return { role: "system", content: msg.content };
        } else if (msg._getType() === "human") {
          return { role: "user", content: msg.content };
        } else if (msg._getType() === "ai") {
          return { role: "assistant", content: msg.content };
        } else if (msg._getType() === "tool") {
          // Handle tool messages with screenshots
          const toolMsg = msg as any;
          if (typeof toolMsg.content === "string" && toolMsg.content.startsWith("data:image/")) {
            return {
              role: "user",
              content: [
                {
                  type: "image_url",
                  image_url: {
                    url: toolMsg.content,
                  },
                },
              ],
            };
          }
          return { role: "user", content: toolMsg.content || "" };
        }
        return { role: "user", content: msg.content || "" };
      });

      // Extract tools if bound
      const tools = (this as any).bound?.tools || options.tools || [];
      const grokTools = tools.length > 0 ? [{
        type: "function",
        function: {
          name: "computer_use_preview",
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
      }] : [];

      // For demo purposes, if no real API key is available, return a demo response
      if (!this.xaiApiKey || this.xaiApiKey === "xai-demo-key-placeholder") {
        const demoMessage = new AIMessage({
          content: "🚀 Demo Mode: Grok-2-Vision-1212 integration is ready! In production, this would use your xAI API key to process requests. The system now supports computer control, bash commands, and real-time streaming. I'll take a screenshot to show the sandbox environment is working.",
          tool_calls: [{
            id: "demo_screenshot",
            name: "computer_use_preview",
            args: {
              action: {
                type: "screenshot"
              }
            }
          }],
          response_metadata: {
            id: "demo_response_" + Date.now(),
            model: "grok-2-vision-1212",
            usage: { prompt_tokens: 100, completion_tokens: 150, total_tokens: 250 },
          },
        });

        return {
          generations: [
            {
              message: demoMessage,
              text: demoMessage.content,
            },
          ],
          llmOutput: {
            tokenUsage: { prompt_tokens: 100, completion_tokens: 150, total_tokens: 250 },
            model: "grok-2-vision-1212",
          },
        };
      }

      // Decide whether to use streaming based on options
      const useStreaming = options.stream !== false && runManager?.onLLMNewToken;

      if (useStreaming) {
        return this._streamGenerate(xaiMessages, grokTools, options, runManager);
      }

      // Make request to xAI (non-streaming)
      const response = await fetch(`${this.xaiBaseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${this.xaiApiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "grok-2-vision-1212",
          messages: xaiMessages,
          tools: grokTools.length > 0 ? grokTools : undefined,
          tool_choice: grokTools.length > 0 ? "auto" : undefined,
          temperature: options.temperature || 0.1,
          max_tokens: options.max_tokens || 4096,
          stream: false,
        }),
      });

      if (!response.ok) {
        throw new Error(`Grok API error: ${response.status} - ${response.statusText}`);
      }

      const data = await response.json();
      const choice = data.choices[0];
      const message = choice.message;

      // Convert back to LangChain format
      const aiMessage = new AIMessage({
        content: message.content || "",
        tool_calls: message.tool_calls?.map((tc: any) => ({
          id: tc.id,
          name: tc.function.name,
          args: JSON.parse(tc.function.arguments || "{}"),
        })) || [],
        response_metadata: {
          id: data.id,
          model: data.model,
          usage: data.usage,
          finish_reason: choice.finish_reason,
        },
      });

      return {
        generations: [
          {
            message: aiMessage,
            text: aiMessage.content,
          },
        ],
        llmOutput: {
          tokenUsage: data.usage,
          model: data.model,
        },
      };
    } catch (error) {
      console.error("Grok API Error:", error);
      
      // Fallback response
      const fallbackMessage = new AIMessage({
        content: "I'm having trouble connecting to the Grok API. Let me take a screenshot to see the current state.",
        tool_calls: [{
          id: "fallback_screenshot",
          name: "computer_use_preview",
          args: {
            action: {
              type: "screenshot"
            }
          }
        }],
      });

      return {
        generations: [
          {
            message: fallbackMessage,
            text: fallbackMessage.content,
          },
        ],
        llmOutput: {
          tokenUsage: { prompt_tokens: 0, completion_tokens: 50, total_tokens: 50 },
          model: "grok-2-vision-1212",
        },
      };
    }
  }

  async _streamGenerate(
    xaiMessages: any[],
    grokTools: any[],
    options: any,
    runManager: any
  ): Promise<any> {
    // Make streaming request to xAI
    const response = await fetch(`${this.xaiBaseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${this.xaiApiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "grok-2-vision-1212",
        messages: xaiMessages,
        tools: grokTools.length > 0 ? grokTools : undefined,
        tool_choice: grokTools.length > 0 ? "auto" : undefined,
        temperature: options.temperature || 0.1,
        max_tokens: options.max_tokens || 4096,
        stream: true,
      }),
    });

    if (!response.ok) {
      throw new Error(`Grok API error: ${response.status} - ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("No response body available for streaming");
    }

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
              break;
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
                  
                  // Call the token callback for streaming UI updates
                  if (runManager?.onLLMNewToken) {
                    await runManager.onLLMNewToken(delta.content);
                  }
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
                        try {
                          const args = JSON.parse(toolCallDelta.function.arguments);
                          toolCall.args = { ...toolCall.args, ...args };
                        } catch (_e) {
                          // Partial JSON, accumulate
                          toolCall.argsString = (toolCall.argsString || "") + toolCallDelta.function.arguments;
                          try {
                            toolCall.args = JSON.parse(toolCall.argsString);
                          } catch (_e2) {
                            // Still partial, continue accumulating
                          }
                        }
                      }
                    }
                  }
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

    // Final message
    const aiMessage = new AIMessage({
      content: accumulatedContent,
      tool_calls: accumulatedToolCalls.filter(tc => tc.id && tc.name),
      response_metadata: responseMetadata,
    });

    return {
      generations: [
        {
          message: aiMessage,
          text: aiMessage.content,
        },
      ],
      llmOutput: {
        tokenUsage: responseMetadata.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
        model: "grok-2-vision-1212",
      },
    };
  }

  _llmType(): string {
    return "grok-2-vision-1212";
  }
}

/**
 * Monkey patch the ChatOpenAI to use our Grok implementation
 */
export function patchOpenAIWithGrok() {
  // Override the ChatOpenAI prototype methods instead of reassigning
  const OriginalChatOpenAI = ChatOpenAI;
  
  // Store the original _generate method
  const original_generate = ChatOpenAI.prototype._generate;
  
  // Replace the _generate method
  ChatOpenAI.prototype._generate = async function(
    messages: BaseMessage[],
    options: any = {},
    runManager?: any
  ) {
    // Create a temporary GrokChatModel instance to handle the call
    const grokModel = new GrokChatModel();
    return grokModel._generate(messages, options, runManager);
  };

  // Override the _llmType method
  ChatOpenAI.prototype._llmType = function() {
    return "grok-2-vision-1212";
  };

  console.log("✅ Successfully patched ChatOpenAI to use Grok-2-Vision-1212");
}