# Gender Identifier

A Streamlit application that provides a cautious, binary facial-presentation estimate from one uploaded or captured image. The app declines to estimate when the face or model result is not reliable enough.

## Live demo

Try the [Gender Identifier Streamlit application](https://gender-identifier.streamlit.app).

## How it works

The application uses RetinaFace through DeepFace to locate and align one face before running DeepFace's pre-trained gender analysis model. It can process portraits and wider photographs when the face contains enough usable detail.

Before displaying an estimate, the reliability layer checks:

- face-detection confidence;
- the face's pixel size in the original image;
- blur, brightness, exposure, and contrast;
- the strongest model score and the margin between both output categories;
- whether the image contains exactly one detected face.

If any check fails, the app explains how to improve the image instead of forcing a prediction. Small images are enlarged for face detection, but the original face size is still assessed so interpolation cannot create false confidence.

## Technology

- DeepFace and RetinaFace
- TensorFlow/Keras
- OpenCV headless
- Pillow and NumPy
- Streamlit

## Run locally

```powershell
git clone https://github.com/Daveokw/Gender-Identifier
cd "Gender-Identifier"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run gender_identifier.py
```

DeepFace downloads its required model weights during the first analysis. The deployment uses `tensorflow-cpu` and headless OpenCV to remain suitable for Streamlit Community Cloud.

Run the lightweight reliability tests with:

```powershell
python -m unittest discover -s tests -v
```

## Important limitations

Model scores are not guarantees or calibrated probabilities. Facial appearance does not determine a person's gender identity, and the model only returns two presentation categories. Results can still be affected by pose, age, lighting, occlusion, image manipulation, and demographic bias.

Do not use this application for identity verification, access decisions, employment, education, healthcare, policing, or any other consequential purpose.
