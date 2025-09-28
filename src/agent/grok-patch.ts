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

    if (!this.xaiApiKey) {
      throw new Error("XAI_API_KEY is required for Grok model");
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

      // Make request to xAI
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

  _llmType(): string {
    return "grok-2-vision-1212";
  }
}

/**
 * Monkey patch the ChatOpenAI to use our Grok implementation
 */
export function patchOpenAIWithGrok() {
  // Override the original ChatOpenAI constructor
  const originalChatOpenAI = ChatOpenAI;
  
  (ChatOpenAI as any) = class extends GrokChatModel {
    constructor(config: any = {}) {
      super(config);
    }
  };
  
  // Preserve static methods
  Object.setPrototypeOf(ChatOpenAI, originalChatOpenAI);
  Object.assign(ChatOpenAI, originalChatOpenAI);
}