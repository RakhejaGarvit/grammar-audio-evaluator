# Audio Transcription and Grammar Scoring Pipeline

This project transcribes audio files using OpenAI's Whisper model and evaluates the grammatical accuracy of the transcription using LanguageTool.

---

## Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Python Dependencies (`requirements.txt`)

```
pandas
numpy
whisper
language-tool-python
torch
ffmpeg-python
```

---

## 🎞️ Install FFmpeg (Required by Whisper)

### For Windows:

1. **Download FFmpeg**  
   Visit: [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)  
   Download the **Release full** ZIP version.

2. **Extract ZIP**  
   Extract to a directory, e.g., `C:\ffmpeg`

3. **Add FFmpeg to System PATH**  
   - Press **Windows + S**, search for `Environment Variables`
   - Click **Environment Variables**
   - Under **System Variables**, find `Path` → click **Edit**
   - Add this new entry:
     ```
     C:\ffmpeg\bin
     ```

4. **Verify Installation**  
   Open Command Prompt and run:

   ```bash
   ffmpeg -version
   ```

   You should see FFmpeg version details if correctly installed.

---

## Notes

- Java is required to use the default LanguageTool. If unavailable, switch to:
  ```python
  language_tool_python.LanguageToolPublicAPI('en-US')
  ```
- Whisper supports different model sizes (`tiny`, `base`, `small`, etc.)

---

## Outputs

- `train_results.csv`: Grammar score predictions for training audio.
- `test_results.csv`: Grammar score predictions for test audio.
- `final_submission.csv`: Output file for submission, matched to sample format.
