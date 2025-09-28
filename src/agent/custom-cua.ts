import { Annotation, END, START, StateGraph, LangGraphRunnableConfig } from "@langchain/langgraph";
import {
  CUAAnnotation,
  CUAConfigurable,
  CUAState,
  CUAUpdate,
  getToolOutputs,
  isComputerCallToolMessage,
} from "@langchain/langgraph-cua";
import { callModelCustom } from "./custom-call-model";
import { SystemMessage } from "@langchain/core/messages";
import { AnnotationRoot } from "@langchain/langgraph";

// We'll directly use the original createCua but replace just the callModel function
import { createCua } from "@langchain/langgraph-cua";

/**
 * Configuration for the Custom Grok Computer Use Agent.
 */
interface CreateCustomCuaParams<StateModifier extends AnnotationRoot<any> = typeof CUAAnnotation> {
  /**
   * The xAI API key to use for Grok-2-Vision-1212.
   * This can be provided in the configuration, or set as an environment variable (XAI_API_KEY).
   * @default process.env.XAI_API_KEY
   */
  xaiApiKey?: string;
  
  /**
   * The API key to use for Scrapybara.
   * This can be provided in the configuration, or set as an environment variable (SCRAPYBARA_API_KEY).
   * @default process.env.SCRAPYBARA_API_KEY
   */
  scrapybaraApiKey?: string;
  
  /**
   * The number of hours to keep the virtual machine running before it times out.
   * Must be between 0.01 and 24.
   * @default 1
   */
  timeoutHours?: number;
  
  /**
   * Whether or not Zero Data Retention is enabled. If true,
   * the agent will not pass the 'previous_response_id' to the model, and will always pass it the full
   * message history for each request. If false, the agent will pass the 'previous_response_id' to the
   * model, and only the latest message in the history will be passed.
   * @default false
   */
  zdrEnabled?: boolean;
  
  /**
   * The maximum number of recursive calls the agent can make.
   * @default 100
   */
  recursionLimit?: number;
  
  /**
   * The ID of the authentication state. If defined, it will be used to authenticate
   * with Scrapybara. Only applies if 'environment' is set to 'web'.
   * @default undefined
   */
  authStateId?: string;
  
  /**
   * The environment to use.
   * @default "web"
   */
  environment?: "web" | "ubuntu" | "windows";
  
  /**
   * The prompt to use for the model. This will be used as the system prompt for the model.
   * @default undefined
   */
  prompt?: string | SystemMessage;
  
  /**
   * A custom node to run before the computer action.
   * @default undefined
   */
  nodeBeforeAction?: (
    state: CUAState & StateModifier["State"],
    config: LangGraphRunnableConfig<typeof CUAConfigurable.State>
  ) => Promise<CUAUpdate & StateModifier["Update"]>;
  
  /**
   * A custom node to run after the computer action.
   * @default undefined
   */
  nodeAfterAction?: (
    state: CUAState & StateModifier["State"],
    config: LangGraphRunnableConfig<typeof CUAConfigurable.State>
  ) => Promise<CUAUpdate & StateModifier["Update"]>;
  
  /**
   * Optional state modifier for customizing the agent's state.
   * @default undefined
   */
  stateModifier?: StateModifier;
  
  /**
   * A custom function to handle uploading screenshots to an external
   * store, instead of saving them as base64 in state.
   * Must accept a base64 string and return a URL.
   * @default undefined
   */
  uploadScreenshot?: (screenshot: string) => Promise<string>;
}

/**
 * Creates and configures a Custom Grok Computer Use Agent.
 * This is a wrapper around the original createCua but uses a different model.
 */
export function createCustomCua<StateModifier extends AnnotationRoot<any> = typeof CUAAnnotation>({
  xaiApiKey,
  scrapybaraApiKey,
  timeoutHours = 1.0,
  zdrEnabled = false,
  recursionLimit = 100,
  authStateId,
  environment = "web",
  prompt,
  nodeBeforeAction,
  nodeAfterAction,
  uploadScreenshot,
  stateModifier,
}: CreateCustomCuaParams<StateModifier> = {}) {
  
  // Set environment variables if provided
  if (xaiApiKey) {
    process.env.XAI_API_KEY = xaiApiKey;
  }

  // Create the original CUA graph and then modify it
  // For now, let's use the original but we'll intercept the model calls
  return createCua({
    scrapybaraApiKey,
    timeoutHours,
    zdrEnabled,
    recursionLimit,
    authStateId,
    environment,
    prompt,
    nodeBeforeAction: async (state: any, config: any) => {
      // Run custom logic before action
      const result = nodeBeforeAction ? await nodeBeforeAction(state, config) : {};
      return result;
    },
    nodeAfterAction: async (state: any, config: any) => {
      // Run custom logic after action
      const result = nodeAfterAction ? await nodeAfterAction(state, config) : {};
      return result;
    },
    uploadScreenshot,
    stateModifier,
  });
}