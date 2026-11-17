import streamlit as st

from page_common import apply_cineverse_style


apply_cineverse_style(
    "Trust and privacy",
    "Understand the trade-offs, safeguards, and limitations behind CineVerse recommendations.",
)

left, right = st.columns(2, gap="large")
with left:
    st.markdown(
        """
        ### Popularity bias
        Popularity-based recommendations can crowd out niche, independent, and international titles.
        The hybrid model combines collaborative and content signals to increase catalog exposure.

        ### Filter bubbles
        Content similarity can reinforce a narrow taste profile. Collaborative signals add controlled
        serendipity by learning from users with related but non-identical preferences.
        """
    )
with right:
    st.markdown(
        """
        ### Privacy
        MovieLens user IDs are anonymous numerical identifiers. CineVerse does not collect names,
        email addresses, or other personal details.

        ### Feedback storage
        Likes and dislikes are saved locally in `feedback.csv`. Streamlit Cloud deployments should
        use durable external storage if feedback must persist across restarts or be shared.
        """
    )

st.info(
    "MovieLens is downloaded on demand under its research terms. The raw dataset is excluded from "
    "the repository, and TMDB posters are optional and require a configured token."
)
