# Leonard Portrait Lab — Phase 1

This is a new project, deliberately named differently from Brushstroke AI V10.

## Run
Create a fresh environment, then:

    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python app.py

The GUI remains simple: upload portrait → GO → watch it sketch and paint.

## What changed
Leonard first builds an imperfect construction sketch from portrait geometry, then paints coarse-to-fine with a fixed red/yellow/blue/black/white palette and three brushes. The current decision-maker is a best-of-N teacher search. This is intentional: it creates the competent teacher needed for behaviour-cloning training rather than wasting compute on random RL.

Outputs: final.png, painting.mp4, portrait_analysis.png, loss.csv.

## Training data
Put public portrait training images in `data/portraits/`. Keep the actual artwork/test portrait out of training. Start with 1,000–5,000 images at 128 px for training experiments.

FFHQ is one possible research dataset, but review its non-commercial/share-alike terms and source-image licences before using it.

## Next training phase
1. Run the teacher over public portraits.
2. Cache state → best-action examples.
3. Train `StrokePolicy` with behaviour cloning.
4. Save `models/portrait_artist.pt`.
5. Use the policy to propose strokes, with a small best-of-N correction search.
6. Only then add parallel RL fine-tuning.

This package intentionally does not train against fake/random action labels.
