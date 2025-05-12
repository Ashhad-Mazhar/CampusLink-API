from llama_cpp import Llama

llm = Llama(
    model_path="./phi2_gguf/phi-2.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,
    use_mlock=True,
)

campuslink_docs = """
WORKFLOW: Join a Society
1. Go to the Societies Page by clicking the button on the sidebar.
2. Search for the society name.
3. Click 'Join Society' on the society page.

WORKFLOW: Leave a Society
1. Navigate to your profile.
2. Click 'My Societies'.
3. Select the society and click 'Leave'.

WORKFLOW: Check Statistics Related to Tasks, Societies, and Events
1. Go to Dashboard by clicking the button on the sidebar.
2. You can now view your statistics on the dashboard.

WORKFLOW: Create New Society
1. Go to Societies Page by clicking the button on the sidebar.
2. Click the 'Register Society' button on the top left.
3. Fill in society name, description, and other details.
4. Submit the request for approval.

WORKFLOW: Chat with other Users
1. Go to Chats Page by clicking the button on the sidebar.
2. Choose the desired society and its channel.
3. Start chatting.

WORKFLOW: Manage Profile, Notification Preferences, 2FA, or Give Feedback on Events
1. Go to Settings Page by clicking the button on the sidebar.
2. Click on the relevant tab to access your desired settings.
3. Edit or enter information as you want.
"""

system_prompt = (
    "You are CampusLink Assistant. You only answer questions using the workflows listed below. "
    "If the question is wildly unrelated or not a greeting, say: 'This might be irrelevant. Please ask CampusLink-related questions.' "
    "Do not make up steps or links. Use only the following workflows:\n\n"
    f"{campuslink_docs}\n\n"
    "Use these exactly. Do not infer extra information. Tell the user the steps they have to take to do the required task. Reference the workflows I have given you.\n\n"
)

def get_answer(user_input):
    prompt = f"{system_prompt}Question: {user_input}\nAnswer:"
    output = llm(prompt, max_tokens=512, stop=["\n\n", "\nQuestion:", "\nAnswer:"])
    raw_text = output["choices"][0]["text"].strip()

    # Optional: Clean up cutoff mid-sentence
    if not raw_text.endswith(".") and "\n" in raw_text:
        raw_text = raw_text[:raw_text.rfind("\n")].strip()

    return raw_text

