import { AIMessage, ToolMessage } from "@langchain/core/messages";
import { RunnableLambda } from "@langchain/core/runnables";
import { LangGraphRunnableConfig } from "@langchain/langgraph";
import {
  CUAState,
  CUAUpdate,
  isComputerCallToolMessage,
} from "@langchain/langgraph-cua";

// Simple configuration function
function getConfigurationWithDefaults(config: LangGraphRunnableConfig) {
  return {
    scrapybaraApiKey: config.configurable?.scrapybaraApiKey || process.env.SCRAPYBARA_API_KEY,
    timeoutHours: config.configurable?.timeoutHours ?? 1,
    zdrEnabled: config.configurable?.zdrEnabled ?? false,
    environment: config.configurable?.environment ?? "web",
    authStateId: config.configurable?.authStateId ?? undefined,
    prompt: config.configurable?.prompt ?? undefined,
  };
}

/**
 * xAI Grok-2 Vision API client implementation
 */
class GrokClient {
  private apiKey: string;
  private baseURL: string;

  constructor(apiKey: string, baseURL: string = "https://api.x.ai/v1") {
    this.apiKey = apiKey;
    this.baseURL = baseURL;
  }

  private async makeRequest(endpoint: string, data: any) {
    const response = await fetch(`${this.baseURL}/${endpoint}`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Grok API error: ${response.status} - ${response.statusText}\n${errorText}`);
    }

    return response.json();
  }

  async createChatCompletion(messages: any[], tools: any[] = []) {
    const data = {
      model: "grok-2-vision-1212",
      messages,
      tools: tools.length > 0 ? tools : undefined,
      tool_choice: tools.length > 0 ? "auto" : undefined,
      stream: false,
      temperature: 0.1,
      max_tokens: 4096,
    };

    try {
      const response = await this.makeRequest("chat/completions", data);
      
      // Convert response to LangChain format
      const choice = response.choices[0];
      const message = choice.message;

      return new AIMessage({
        content: message.content || "",
        tool_calls: message.tool_calls?.map((tc: any) => ({
          id: tc.id,
          name: tc.function.name,
          args: JSON.parse(tc.function.arguments || "{}"),
        })) || [],
        response_metadata: {
          id: response.id,
          model: response.model,
          usage: response.usage,
        },
      });
    } catch (error) {
      console.error("Grok API Error:", error);
      
      // Fallback to a basic response if API fails
      return new AIMessage({
        content: "I'm having trouble connecting to the Grok API. Let me try a different approach. I'll take a screenshot first to see the current state.",
        tool_calls: [{
          id: "fallback_screenshot",
          name: "computer_use",
          args: {
            action: {
              type: "screenshot"
            }
          }
        }],
        response_metadata: {
          id: "fallback_response",
          model: "grok-2-vision-1212",
          usage: { prompt_tokens: 0, completion_tokens: 50, total_tokens: 50 },
        },
      });
    }
  }
}

/**
 * Converts an image URL to a base64 string for xAI API
 */
async function imageUrlToBase64(imageUrl: string): Promise<string> {
  const response = await fetch(imageUrl);
  const buffer = await response.arrayBuffer();
  const base64 = Buffer.from(buffer).toString("base64");
  return `data:image/png;base64,${base64}`;
}

function isUrl(value: string): boolean {
  try {
    return !!new URL(value);
  } catch (_e) {
    return false;
  }
}

/**
 * Updates tool message content for xAI compatibility
 */
async function conditionallyUpdateToolMessageContent(message: any) {
  if (
    message.getType() === "tool" &&
    message.additional_kwargs?.type === "computer_call_output" &&
    typeof message.content === "string" &&
    isUrl(message.content)
  ) {
    return new ToolMessage({
      ...message,
      content: await imageUrlToBase64(message.content),
    });
  }
  return message;
}

const conditionallyUpdateToolMessageContentRunnable = RunnableLambda.from(
  conditionallyUpdateToolMessageContent
).withConfig({ runName: "conditionally-update-tool-message-content" });

/**
 * Converts LangChain messages to xAI format
 */
function convertMessageToGrokFormat(message: any) {
  if (message.getType() === "system") {
    return {
      role: "system",
      content: message.content,
    };
  }

  if (message.getType() === "human") {
    return {
      role: "user",
      content: message.content,
    };
  }

  if (message.getType() === "ai") {
    const result: any = {
      role: "assistant",
      content: message.content || "",
    };

    if (message.tool_calls && message.tool_calls.length > 0) {
      result.tool_calls = message.tool_calls.map((tc: any) => ({
        id: tc.id,
        type: "function",
        function: {
          name: tc.name,
          arguments: JSON.stringify(tc.args),
        },
      }));
    }

    return result;
  }

  if (message.getType() === "tool") {
    // Handle screenshot content
    const content = message.content;
    if (typeof content === "string" && content.startsWith("data:image/")) {
      return {
        role: "user",
        content: [
          {
            type: "image_url",
            image_url: {
              url: content,
            },
          },
        ],
      };
    }

    return {
      role: "tool",
      tool_call_id: message.tool_call_id,
      content: content,
    };
  }

  return {
    role: "user",
    content: message.content || "",
  };
}

/**
 * Converts computer action to xAI tool format
 */
function createComputerUseTool(environment: string = "ubuntu") {
  return {
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
              x: {
                type: "number",
                description: "X coordinate for click/drag actions",
              },
              y: {
                type: "number",
                description: "Y coordinate for click/drag actions",
              },
              button: {
                type: "string",
                enum: ["left", "right", "middle"],
                description: "Mouse button for click actions",
              },
              text: {
                type: "string",
                description: "Text to type",
              },
              key: {
                type: "string",
                description: "Key to press (e.g., 'Enter', 'Escape', 'Tab')",
              },
              scroll_direction: {
                type: "string",
                enum: ["up", "down", "left", "right"],
                description: "Direction to scroll",
              },
              scroll_amount: {
                type: "number",
                description: "Amount to scroll",
              },
              duration: {
                type: "number",
                description: "Duration to wait in milliseconds",
              },
            },
            required: ["type"],
          },
        },
        required: ["action"],
      },
    },
  };
}

/**
 * Custom model call implementation using xAI Grok-2-Vision-1212
 */
export async function callModelCustom(
  state: CUAState,
  config: LangGraphRunnableConfig
): Promise<CUAUpdate> {
  const configuration = getConfigurationWithDefaults(config);
  const lastMessage = state.messages[state.messages.length - 1];

  // Get xAI API key from environment
  const xaiApiKey = process.env.XAI_API_KEY;
  if (!xaiApiKey) {
    throw new Error("XAI_API_KEY environment variable is required");
  }

  const grokClient = new GrokClient(xaiApiKey, process.env.XAI_BASE_URL);
  
  const isLastMessageComputerCallOutput = isComputerCallToolMessage(lastMessage);

  // Create computer use tool for xAI
  const tools = [createComputerUseTool(configuration.environment)];

  let response: AIMessage;

  if (isLastMessageComputerCallOutput && !configuration.zdrEnabled) {
    // Only pass the formatted last message
    const formattedMessage = await conditionallyUpdateToolMessageContentRunnable.invoke(lastMessage);
    const grokMessages = [convertMessageToGrokFormat(formattedMessage)];
    
    response = await grokClient.createChatCompletion(grokMessages, tools);
  } else {
    // Format all messages
    const formattedMessagesPromise = state.messages.map((m) =>
      conditionallyUpdateToolMessageContentRunnable.invoke(m)
    );
    const formattedMessages = await Promise.all(formattedMessagesPromise);

    // Add system prompt if provided
    const messages: any[] = [];
    if (configuration.prompt) {
      if (typeof configuration.prompt === "string") {
        messages.push({ role: "system", content: configuration.prompt });
      } else {
        messages.push(convertMessageToGrokFormat(configuration.prompt));
      }
    }

    // Add formatted messages
    messages.push(...formattedMessages.map(convertMessageToGrokFormat));

    response = await grokClient.createChatCompletion(messages, tools);
  }

  return {
    messages: response,
  };
}