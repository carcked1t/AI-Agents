import streamlit as st
from youtubetranscripter import get_transcript, generate_social_media_content

st.set_page_config(page_title="Social Media Content Generator", page_icon="📱")

st.title("📱 Social Media Content Generator")
st.write("Generate platform-specific content from any YouTube video transcript.")

# --- Inputs ---
video_id = st.text_input("YouTube Video ID", placeholder="e.g., Zxs7Rf2rWxc")

query = st.text_area(
    "Your Query",
    placeholder="Explain the type of content you want (e.g., Generate a LinkedIn post summarizing key ideas)...",
    height=120
)

platforms = st.multiselect(
    "Select Social Media Platforms",
    ["LinkedIn", "Instagram", "Twitter/X", "Facebook", "YouTube Community Post", "Reddit", "Blog"],
    default=["LinkedIn"]
)

submit = st.button("Generate Content")

# --- Processing ---
if submit:
    if not video_id:
        st.error("Please enter a YouTube video ID.")
    elif not query:
        st.error("Please enter a query.")
    elif not platforms:
        st.error("Please select at least one platform.")
    else:
        with st.spinner("Fetching transcript..."):
            transcript = get_transcript(video_id)

        if transcript.strip() == "" or transcript == "Transcript not available.":
            st.error("❌ Transcript not available for this video.")
        else:
            st.success("Transcript retrieved!")

            st.markdown("---")
            st.header(" Generated Content")

            for platform in platforms:
                full_prompt = f"{query}\n\nHere is the video transcript:\n{transcript}"

                with st.spinner(f"Generating {platform} content..."):
                    result = generate_social_media_content(full_prompt, platform)

                st.subheader(f"➡️ {platform}")
                st.write(result)
                st.markdown("---")
