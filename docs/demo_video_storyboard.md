# 📹 3-Minute Backup Demo Video Storyboard & Recording Script

**Project:** Hybrid Movie Recommendation System  
**Owner:** Seat 13 (Deployment Specialist)  
**Format:** 1080p Screen Recording with clear voiceover narrative  
**Duration:** Exactly 3 Minutes (180 Seconds)  
**Output Target:** `Group##_MovieRecommender_Demo.mp4`

---

## Technical Recording Setup
1. **Screen Resolution:** 1920 $\times$ 1080 (16:9 widescreen). Browser zoomed to 100% or 110% for crisp typography.
2. **Tools:** OBS Studio, QuickTime, or Windows Game Bar (Win+G). Audio recorded via dedicated microphone with background noise suppression.
3. **Environment:** Local Streamlit server running via `streamlit run app/app.py` on `localhost:8501`.
4. **Clean State:** Empty or sample `feedback.csv` so newly clicked feedback updates live on screen.

---

## Scene-by-Scene Storyboard & Voiceover Script

### Segment 1: Problem Statement & System Introduction (0:00 – 0:30 | 30 Seconds)
- **Visual:**
  - Starts on title slide / Streamlit header banner: *"🎬 CineMatch: Hybrid Movie Recommender"*.
  - Camera pans smoothly down to the project description and architecture overview.
- **Voiceover:**
  > "Welcome. In this demonstration, we present our end-to-end Hybrid Movie Recommendation System developed for the Recommender Systems Practical Study at JKUAT.
  > Modern recommender engines face three critical obstacles: extreme interaction sparsity, the cold-start problem for new visitors, and popularity bias.
  > To solve this, our system unifies User-based and Item-based Collaborative Filtering, SVD Matrix Factorization, Content-based TF-IDF, and PyTorch Neural Collaborative Filtering into a tuned Weighted Hybrid, deployed here on Streamlit."

---

### Segment 2: Existing User Personalized Recommendations & Feedback Logging (0:30 – 2:00 | 90 Seconds)
- **Visual:**
  - Cursor selects **User ID #10** from the user dropdown.
  - Expands the "👤 View Taste Profile for User #10" panel showing 140 ratings, favorite genres (Action, Thriller, Crime), and top-rated movies (*Pulp Fiction*, *The Matrix*).
  - Selects the **⭐ Weighted Hybrid** model from the model dropdown.
  - Ten recommendation cards appear in a 2-column grid (*Heat*, *Fargo*, *Reservoir Dogs*, *Fight Club*).
  - Cursor clicks **"👍 Yes"** on recommendation #1 (*Heat*). A green confirmation toast appears: *"Feedback logged!"*.
  - Cursor switches model to **"🔥 Popularity Baseline"** to show the contrast: the list collapses to generic movies (*Forrest Gump*, *Shawshank Redemption*).
  - Cursor switches to **"🔄 Compare All Models Side-by-Side"** expander, revealing the matrix comparing ranks across all 8 models simultaneously.
- **Voiceover:**
  > "Let's examine how the system recommends for an existing user. We select User #10.
  > Opening their taste profile, we see they are an avid fan of 90s action, crime, and neo-noir films with over 140 rated titles.
  > Under our Weighted Hybrid model, the engine blends user-similarity and item-similarity signals, serving highly personalized titles like *Heat* and *Fargo*.
  > When the user interacts with a recommendation, they can click 'Thumbs Up' or 'Thumbs Down'. Notice how clicking 'Thumbs Up' on *Heat* immediately logs an implicit positive reaction to our local `feedback.csv` file.
  > Switching to the Popularity baseline illustrates the limitation of non-personalized models—it simply serves the most-rated movies regardless of user taste. Our side-by-side comparison table clearly displays how each algorithm ranks candidate titles differently."

---

### Segment 3: Cold-Start Fallback Demonstration (2:00 – 2:30 | 30 Seconds)
- **Visual:**
  - Cursor clicks on the **"❄️ Cold-Start Onboarding"** tab.
  - Selects the pre-configured persona **"Alex Vance - The Sci-Fi & Cyberpunk Fan"**.
  - Shows the profile box with seed genres (*Sci-Fi, Action, Thriller*) and seed movies (*The Matrix, Terminator 2*).
  - Clicks the big blue button: **"🚀 Generate Cold-Start Recommendations"**.
  - Instant top-10 content-based recommendations appear (*Blade Runner*, *Star Wars*, *Aliens*, *Twelve Monkeys*) with 100% genre alignment.
  - Clicks 👍 on the top recommendation.
- **Voiceover:**
  > "Now, what happens when a brand new user visits the platform? Collaborative filtering completely breaks down due to lack of co-rating history.
  > We solve this using our Content-based TF-IDF fallback engine. Here we select a demo persona—Alex, a Sci-Fi and Cyberpunk enthusiast.
  > Clicking 'Generate Recommendations', our TF-IDF model evaluates over 9,700 movie vector representations, matching plot tags and genres in under 50 milliseconds.
  > The result is an instant, highly relevant top-10 list featuring *Blade Runner* and *Star Wars*, with full feedback capture."

---

### Segment 4: Model Benchmarks, Ethics & Conclusion (2:30 – 3:00 | 30 Seconds)
- **Visual:**
  - Cursor clicks on the **"📊 Benchmarks & Evaluation"** tab.
  - Shows the official metrics table: Weighted Hybrid winning with **Recall@10 = 0.0964** and **NDCG@10 = 0.0521**, SVD with **RMSE = 0.8831**.
  - Highlights the **Figure 1** metrics comparison bar chart and **Figure 2** catalog coverage chart.
  - Briefly clicks the **"⚖️ Ethics & Privacy"** tab to show the catalog coverage analysis (hybrid expanding coverage from 1.25% to 4.12%).
  - Returns to header.
- **Voiceover:**
  > "Finally, our Benchmarks dashboard summarizes our leave-one-out evaluation.
  > The Weighted Hybrid emerges as our top performer, achieving a Recall@10 of 0.0964 and NDCG@10 of 0.0521, outperforming popularity, pure collaborative, and deep learning baselines.
  > Concurrently, our ethics audit demonstrates that the hybrid triples catalog coverage over the popularity baseline, mitigating systemic bias while maintaining complete user privacy.
  > All source code, data splits, and model weights are fully documented and reproducible in our repository. Thank you."

---

## Post-Recording Verification Checklist
- [x] Video length is between 2:50 and 3:05.
- [x] Audio voiceover is loud, clear, and synchronized with screen cursor actions.
- [x] 1080p resolution with no personal desktop notifications visible.
- [x] File named `Group##_MovieRecommender_Demo.mp4` and saved into shared Drive folder.
