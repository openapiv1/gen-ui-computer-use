import {
  createCua,
  CUAAnnotation,
  CUAState,
  CUAUpdate,
  getToolOutputs,
  isComputerCallToolMessage,
} from "@langchain/langgraph-cua";
import {
  typedUi,
  uiMessageReducer,
} from "@langchain/langgraph-sdk/react-ui/server";
import type ComponentMap from "./ui/index";
import { Annotation, LangGraphRunnableConfig } from "@langchain/langgraph";
import {
  type UIMessage,
  type RemoveUIMessage,
} from "@langchain/langgraph-sdk/react-ui";
import { createClient } from "@supabase/supabase-js";
import { v4 as uuidv4 } from "uuid";

// Patch ChatOpenAI to use Grok-2-Vision-1212 instead
import { patchOpenAIWithGrok } from "./grok-patch";

// Apply the patch before any other imports that might use ChatOpenAI
patchOpenAIWithGrok();

const GraphAnnotation = Annotation.Root({
  ...CUAAnnotation.spec,
  ui: Annotation<
    UIMessage[],
    UIMessage | RemoveUIMessage | (UIMessage | RemoveUIMessage)[]
  >({ default: () => [], reducer: uiMessageReducer }),
});

type GraphState = typeof GraphAnnotation.State & CUAState;

function convertBase64ToBlob(screenshot: string): Blob | null {
  let base64Data = screenshot;
  if (screenshot.startsWith("data:image/png;base64,")) {
    base64Data = screenshot.slice("data:image/png;base64,".length);
  }

  try {
    // Decode the base64 string
    const byteCharacters = atob(base64Data);
    // Create an array of byte values
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    // Convert to a typed array
    const byteArray = new Uint8Array(byteNumbers);
    // Create and return the Blob
    return new Blob([byteArray], { type: "image/png" });
  } catch (error) {
    console.error("Failed to convert base64 to Blob:", error);
    // Return null or handle the error as appropriate
    return null;
  }
}

async function uploadScreenshot(screenshot: string): Promise<string> {
  const supabaseUrl = process.env.SUPABASE_URL;
  const supabaseApiKey = process.env.SUPABASE_API_KEY;
  if (!supabaseUrl || !supabaseApiKey) {
    throw new Error("Missing Supabase credentials");
  }

  const bucketName = "cua-screenshots";

  const client = createClient(supabaseUrl, supabaseApiKey);

  const fileName = `${uuidv4()}.png`;

  const blob = convertBase64ToBlob(screenshot);
  if (!blob) {
    console.error("Failed to convert screenshot to blob. Aborting upload.");
    throw new Error("Failed to process screenshot for upload.");
  }

  const { error: uploadError } = await client.storage
    .from(bucketName)
    .upload(fileName, blob, {
      contentType: "image/png",
      duplex: "half",
      upsert: false,
    });

  if (uploadError) {
    throw new Error(`Failed to upload screenshot: ${uploadError.message}`);
  }

  const expiresIn = 60 * 60 * 24 * 90; // 90 days

  const { data, error: signedUrlError } = await client.storage
    .from(bucketName)
    .createSignedUrl(fileName, expiresIn);

  if (signedUrlError) {
    throw new Error(`Failed to create signed URL: ${signedUrlError.message}`);
  }

  return data.signedUrl;
}

async function beforeNode(
  state: GraphState,
  config: LangGraphRunnableConfig,
): Promise<CUAUpdate> {
  const ui = typedUi<typeof ComponentMap>(config);
  const lastMessage = state.messages[state.messages.length - 1];
  const toolCalls = getToolOutputs(lastMessage);

  const renderVMButton = state.ui.find(
    (message) => message.name === "render-vm-button",
  );
  const instanceFrame = state.ui.find((message) => message.name === "instance");

  if (state.instanceId && state.streamUrl) {
    if (!instanceFrame) {
      ui.push(
        {
          name: "instance",
          props: {
            instanceId: state.instanceId,
            streamUrl: state.streamUrl,
          },
        },
        {
          message: lastMessage,
        },
      );
    } else if (
      instanceFrame.props.instanceId !== state.instanceId ||
      instanceFrame.props.streamUrl !== state.streamUrl
    ) {
      // First, remove the existing instance frame
      ui.delete(instanceFrame.id);
      // Then, add the new instance frame
      ui.push(
        {
          name: "instance",
          props: {
            instanceId: state.instanceId,
            streamUrl: state.streamUrl,
          },
        },
        {
          message: lastMessage,
        },
      );
    }

    // Check if there are any UI messages in state. If there are not, we can assume the UI button for rendering the VM is not visible
    if (!renderVMButton) {
      ui.push(
        {
          name: "render-vm-button",
          props: {},
          metadata: {
            instanceId: state.instanceId,
            streamUrl: state.streamUrl,
          },
        },
        {
          message: lastMessage,
        },
      );
    } else if (
      renderVMButton.metadata.instanceId !== state.instanceId ||
      renderVMButton.metadata.streamUrl !== state.streamUrl
    ) {
      // There is a render VM button, but it's tied to an old instance. Push a new one.
      // This will improve the UX so the render button is "closer" in the UI to the new request
      ui.delete(renderVMButton.id);
      ui.push(
        {
          name: "render-vm-button",
          props: {},
          metadata: {
            instanceId: state.instanceId,
            streamUrl: state.streamUrl,
          },
        },
        {
          message: lastMessage,
        },
      );
    }
  }

  if (toolCalls?.length) {
    toolCalls.map((tc) => {
      ui.push(
        {
          name: "computer-use-tool-call",
          props: {
            toolCallId: tc.id,
            action: tc.action,
          },
        },
        {
          message: lastMessage,
        },
      );
    });
  }

  return {};
}

async function afterNode(
  state: GraphState,
  config: LangGraphRunnableConfig,
): Promise<CUAUpdate> {
  const ui = typedUi<typeof ComponentMap>(config);
  const lastMessage = state.messages[state.messages.length - 1];
  if (isComputerCallToolMessage(lastMessage)) {
    ui.push(
      {
        name: "computer-use-tool-output",
        props: {
          toolCallId: lastMessage.tool_call_id,
          screenshot: lastMessage.content as string,
        },
      },
      {
        message: lastMessage,
      },
    );
  }

  return {};
}

export const graph = createCua({
  nodeBeforeAction: beforeNode,
  nodeAfterAction: afterNode,
  stateModifier: GraphAnnotation,
  recursionLimit: 150,
  timeoutHours: 0.1,
  uploadScreenshot,
  environment: "ubuntu", // Explicitly set environment for better computer control
  prompt: `You are a computer use assistant powered by grok-2-vision-1212. You can:
- Control the computer through clicking, typing, and other actions
- Take screenshots to see the current state
- Use bash commands to execute shell operations
- Navigate and interact with the graphical interface

Always take a screenshot first to see the current state before taking any action.
Be precise with coordinates and take time to analyze the screen before acting.
If a task requires multiple steps, break it down and execute them sequentially.`,
});
