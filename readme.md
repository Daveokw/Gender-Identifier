# Gender Identifier

A Streamlit application that provides a binary facial-presentation estimate from one uploaded or captured image. The app asks for a clearer image when it cannot obtain enough usable facial detail.

## Live demo

Try the [Gender Identifier Streamlit application](https://gender-identifier.streamlit.app).

## How it works

The application uses a high-accuracy RetinaFace ResNet-34 detector to locate one face, an eDifFIQA model to assess whether the face contains usable visual detail, and the balanced FairFace classifier to produce the estimate. All three models run locally in the Streamlit process through UniFace and ONNX Runtime, without a third-party prediction API.

Before displaying an estimate, the image-quality layer checks:

- the face's pixel size in the original image;
- blur, brightness, exposure, and contrast;
- whether the image contains exactly one detected face.

If any check fails, the app explains how to improve the image instead of forcing a prediction. Small images are enlarged for face detection, but the original face size is still assessed so interpolation cannot create missing detail.

## Technology

- FairFace, RetinaFace ResNet-34, and eDifFIQA-T
- UniFace and ONNX Runtime
- OpenCV
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

UniFace downloads and verifies the free RetinaFace, eDifFIQA, and FairFace weights during the first analysis. Later analyses reuse the cached ONNX models.

Run the lightweight reliability tests with:

```powershell
python -m unittest discover -s tests -v
```

## Important limitations

Facial appearance does not determine a person's gender identity, and the model only returns two presentation categories. Results can still be affected by pose, age, lighting, occlusion, image manipulation, and demographic bias.

Do not use this application for identity verification, access decisions, employment, education, healthcare, policing, or any other consequential purpose.

## Model attribution

The application uses [UniFace](https://github.com/yakhyo/uniface), released under the MIT licence, and [FairFace](https://github.com/dchen236/FairFace) model weights distributed under CC BY 4.0. FairFace was introduced by Kimmo Kärkkäinen and Jungseock Joo in *FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation*.
