let mediaRecorder, audioChunks, audioBlob, stream, audioRecorded;
const ctx = new AudioContext();
let currentAudioForPlaying;
let lettersOfWordAreCorrect = [];

const page_title = "AI Pronunciation Trainer";
const accuracy_colors = ["green", "orange", "red"];
let badScoreThreshold = 30;
let mediumScoreThreshold = 70;
let currentSample = 0;
let currentScore = 0;
let playAnswerSounds = true;
let isNativeSelectedForPlayback = true;
let isRecording = false;
let currentSoundRecorded = false;
let currentText, currentIpa, real_transcripts_ipa, matched_transcripts_ipa;
let wordCategories;
let startTime, endTime;

const AILanguage = "en";
const soundsPath = "/static";
let soundFileGood = null;
let soundFileOkay = null;
let soundFileBad = null;

const synth = window.speechSynthesis;
let voice_synth = null;

const setEnglishVoice = () => {
  const voices = synth.getVoices();
  const languageIdentifier = "en";
  const languageName = "Daniel";

  for (let idx = 0; idx < voices.length; idx++) {
    if (
      voices[idx].lang.slice(0, 2) === languageIdentifier &&
      voices[idx].name === languageName
    ) {
      voice_synth = voices[idx];
      return;
    }
  }
  for (let idx = 0; idx < voices.length; idx++) {
    if (voices[idx].lang.slice(0, 2) === languageIdentifier) {
      voice_synth = voices[idx];
      return;
    }
  }
};

if (synth.onvoiceschanged !== undefined) {
  synth.onvoiceschanged = setEnglishVoice;
}

const unblockUI = () => {
  document.getElementById("recordAudio").classList.remove("disabled");
  document.getElementById("playSampleAudio").classList.remove("disabled");
  document.getElementById("buttonNext").onclick = () => getNextSample();
  document.getElementById("nextButtonDiv").classList.remove("disabled");
  document.getElementById("original_script").classList.remove("disabled");
  document.getElementById("buttonNext").style["background-color"] = "#58636d";

  if (currentSoundRecorded) {
    document.getElementById("playRecordedAudio").classList.remove("disabled");
  }
};

const blockUI = () => {
  document.getElementById("recordAudio").classList.add("disabled");
  document.getElementById("playSampleAudio").classList.add("disabled");
  document.getElementById("buttonNext").onclick = null;
  document.getElementById("original_script").classList.add("disabled");
  document.getElementById("playRecordedAudio").classList.add("disabled");
  document.getElementById("buttonNext").style["background-color"] = "#adadad";
};

const UIError = () => {
  blockUI();
  document.getElementById("buttonNext").onclick = () => getNextSample();
  document.getElementById("buttonNext").style["background-color"] = "#58636d";
  document.getElementById("recorded_ipa_script").innerHTML = "";
  document.getElementById("single_word_ipa_pair").innerHTML = "Error";
  document.getElementById("ipa_script").innerHTML = "Error";
  document.getElementById("main_title").innerHTML = "Server Error";
  document.getElementById("original_script").innerHTML =
    "A server error occurred. Please try generating a new sample in a few seconds.";
};

const UINotSupported = () => {
  unblockUI();
  document.getElementById("main_title").innerHTML = "Browser unsupported";
};

const UIRecordingError = () => {
  unblockUI();
  document.getElementById("main_title").innerHTML =
    "Recording error, please try again or restart page.";
  startMediaDevice();
};

function updateScore(currentPronunciationScore) {
  if (isNaN(currentPronunciationScore)) return;
  currentScore += currentPronunciationScore;
  currentScore = Math.round(currentScore);
}

const cacheSoundFiles = async () => {
  soundFileGood = await fetch(soundsPath + "/ASR_good.wav")
    .then((data) => data.arrayBuffer())
    .then((arrayBuffer) => ctx.decodeAudioData(arrayBuffer));
  soundFileOkay = await fetch(soundsPath + "/ASR_okay.wav")
    .then((data) => data.arrayBuffer())
    .then((arrayBuffer) => ctx.decodeAudioData(arrayBuffer));
  soundFileBad = await fetch(soundsPath + "/ASR_bad.wav")
    .then((data) => data.arrayBuffer())
    .then((arrayBuffer) => ctx.decodeAudioData(arrayBuffer));
};

const getNextSample = async () => {
  blockUI();
  if (soundFileBad == null) cacheSoundFiles();
  updateScore(
    parseFloat(document.getElementById("pronunciation_accuracy").innerHTML)
  );
  document.getElementById("main_title").innerHTML = "Processing new sample...";

  // Xác định endpoint cần gọi dựa trên radio button
  let endpointUrl = "/getExampleWord";
  if (document.getElementById("modeSentence").checked) {
    endpointUrl = "/getExampleSentence";
  }

  try {
    // Gọi đến endpoint đã xác định
    await fetch(endpointUrl, {
      method: "post",
      headers: { "Content-Type": "application/json" },
      // Không cần gửi body nữa
      body: JSON.stringify({}),
    })
      .then((res) => res.json())
      .then((data) => {
        let doc = document.getElementById("original_script");
        currentText = data.real_transcript;
        doc.innerHTML = currentText;
        currentIpa = data.ipa_transcript;
        let doc_ipa = document.getElementById("ipa_script");
        doc_ipa.innerHTML = "/ " + currentIpa + " /";
        document.getElementById("recorded_ipa_script").innerHTML = "";
        document.getElementById("pronunciation_accuracy").innerHTML = "";
        document.getElementById("single_word_ipa_pair").innerHTML =
          "Reference | Spoken";
        document.getElementById("section_accuracy").innerHTML =
          "| Score: " +
          currentScore.toString() +
          " - (" +
          currentSample.toString() +
          ")";
        currentSample += 1;
        document.getElementById("main_title").innerHTML = page_title;
        document.getElementById("translated_script").innerHTML =
          data.transcript_translation;
        currentSoundRecorded = false;
        unblockUI();
        document.getElementById("playRecordedAudio").classList.add("disabled");
      });
  } catch {
    UIError();
  }
};

const updateRecordingState = async () => {
  if (isRecording) {
    stopRecording();
  } else {
    recordSample();
  }
};

const generateWordModal = (word_idx) => {
  document.getElementById("single_word_ipa_pair").innerHTML =
    wrapWordForPlayingLink(
      real_transcripts_ipa[word_idx],
      word_idx,
      false,
      "black"
    ) +
    " | " +
    wrapWordForPlayingLink(
      matched_transcripts_ipa[word_idx],
      word_idx,
      true,
      accuracy_colors[parseInt(wordCategories[word_idx])]
    );
};

const recordSample = async () => {
  document.getElementById("main_title").innerHTML =
    "Recording... click again when done speaking";
  document.getElementById("recordIcon").innerHTML = "pause_presentation";
  blockUI();
  document.getElementById("recordAudio").classList.remove("disabled");
  audioChunks = [];
  isRecording = true;
  mediaRecorder.start();
};

const mediaStreamConstraints = {
  audio: { channelCount: 1, sampleRate: 48000 },
};

const startMediaDevice = () => {
  navigator.mediaDevices
    .getUserMedia(mediaStreamConstraints)
    .then((_stream) => {
      stream = _stream;
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (event) => audioChunks.push(event.data);
      mediaRecorder.onstop = async () => {
        document.getElementById("recordIcon").innerHTML = "mic";
        blockUI();
        audioBlob = new Blob(audioChunks, { type: "audio/ogg;" });
        let audioUrl = URL.createObjectURL(audioBlob);
        audioRecorded = new Audio(audioUrl);
        let audioBase64 = await convertBlobToBase64(audioBlob);

        if (audioBase64.length < 6) {
          setTimeout(UIRecordingError, 50);
          return;
        }
        try {
          let text = document.getElementById("original_script").innerHTML;
          text = text
            .replace(/<[^>]*>?/gm, "")
            .trim()
            .replace(/\s\s+/g, " ");
          currentText = [text];

          await fetch("/GetAccuracyFromRecordedAudio", {
            method: "post",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              title: currentText[0],
              base64Audio: audioBase64,
              language: AILanguage,
            }),
          })
            .then((res) => res.json())
            .then((data) => {
              if (playAnswerSounds)
                playSoundForAnswerAccuracy(
                  parseFloat(data.pronunciation_accuracy)
                );
              document.getElementById("recorded_ipa_script").innerHTML =
                "/ " + data.ipa_transcript + " /";
              document.getElementById("recordAudio").classList.add("disabled");
              document.getElementById("main_title").innerHTML = page_title;
              document.getElementById("pronunciation_accuracy").innerHTML =
                data.pronunciation_accuracy + "%";
              document.getElementById("ipa_script").innerHTML =
                data.real_transcripts_ipa;
              lettersOfWordAreCorrect =
                data.is_letter_correct_all_words.split(" ");
              startTime = data.start_time;
              endTime = data.end_time;
              real_transcripts_ipa = data.real_transcripts_ipa.split(" ");
              matched_transcripts_ipa = data.matched_transcripts_ipa.split(" ");
              wordCategories = data.pair_accuracy_category.split(" ");
              let currentTextWords = currentText[0].split(" ");
              let coloredWords = "";
              for (
                let word_idx = 0;
                word_idx < currentTextWords.length;
                word_idx++
              ) {
                let wordTemp = "";
                for (
                  let letter_idx = 0;
                  letter_idx < currentTextWords[word_idx].length;
                  letter_idx++
                ) {
                  let letter_is_correct =
                    lettersOfWordAreCorrect[word_idx][letter_idx] === "1";
                  let color_letter = letter_is_correct ? "green" : "red";
                  wordTemp +=
                    "<font color=" +
                    color_letter +
                    ">" +
                    currentTextWords[word_idx][letter_idx] +
                    "</font>";
                }
                coloredWords +=
                  " " + wrapWordForIndividualPlayback(wordTemp, word_idx);
              }
              document.getElementById("original_script").innerHTML =
                coloredWords;
              currentSoundRecorded = true;
              unblockUI();
              document
                .getElementById("playRecordedAudio")
                .classList.remove("disabled");
            });
        } catch {
          UIError();
        }
      };
    });
};
startMediaDevice();

const playSoundForAnswerAccuracy = async (accuracy) => {
  if (accuracy < mediumScoreThreshold) {
    currentAudioForPlaying =
      accuracy < badScoreThreshold ? soundFileBad : soundFileOkay;
  } else {
    currentAudioForPlaying = soundFileGood;
  }
  playback();
};

const playAudio = async () => {
  document.getElementById("main_title").innerHTML = "Generating sound...";
  playWithBrowserApi(currentText[0]);
};

function playback() {
  const playSound = ctx.createBufferSource();
  playSound.buffer = currentAudioForPlaying;
  playSound.connect(ctx.destination);
  playSound.start(ctx.currentTime);
}

const playRecording = async (start = null, end = null) => {
  blockUI();
  try {
    if (start == null || end == null) {
      audioRecorded.onended = () => {
        audioRecorded.currentTime = 0;
        unblockUI();
        document.getElementById("main_title").innerHTML =
          "Recorded Sound was played";
      };
      await audioRecorded.play();
    } else {
      audioRecorded.currentTime = start;
      audioRecorded.play();
      let durationInMs = Math.round((end - start) * 1000);
      setTimeout(() => {
        unblockUI();
        audioRecorded.pause();
        audioRecorded.currentTime = 0;
        document.getElementById("main_title").innerHTML =
          "Recorded Sound was played";
      }, durationInMs);
    }
  } catch {
    UINotSupported();
  }
};

const playNativeAndRecordedWord = async (word_idx) => {
  if (isNativeSelectedForPlayback) {
    playCurrentWord(word_idx);
  } else {
    playRecordedWord(word_idx);
  }
  isNativeSelectedForPlayback = !isNativeSelectedForPlayback;
};

const stopRecording = () => {
  isRecording = false;
  mediaRecorder.stop();
  document.getElementById("main_title").innerHTML = "Processing audio...";
};

const playCurrentWord = async (word_idx) => {
  document.getElementById("main_title").innerHTML = "Generating word...";
  playWithBrowserApi(currentText[0].split(" ")[word_idx]);
};

const playWithBrowserApi = (text) => {
  if (!voice_synth) setEnglishVoice();
  if (voice_synth) {
    blockUI();
    let utterThis = new SpeechSynthesisUtterance(text);
    utterThis.voice = voice_synth;
    utterThis.rate = 0.7;
    utterThis.onend = () => {
      unblockUI();
      document.getElementById("main_title").innerHTML =
        "Sample sound was played";
    };
    synth.speak(utterThis);
  } else {
    UINotSupported();
  }
};

const playRecordedWord = (word_idx) => {
  let wordStartTime = parseFloat(startTime.split(" ")[word_idx]);
  let wordEndTime = parseFloat(endTime.split(" ")[word_idx]);
  playRecording(wordStartTime, wordEndTime);
};

const convertBlobToBase64 = async (blob) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(blob);
    reader.onload = () => resolve(reader.result);
    reader.onerror = (error) => reject(error);
  });

const wrapWordForPlayingLink = (
  word,
  word_idx,
  isFromRecording,
  word_accuracy_color
) => {
  const functionCall = isFromRecording
    ? `playRecordedWord(${word_idx})`
    : `playCurrentWord(${word_idx})`;
  return `<a style="white-space:nowrap; color:${word_accuracy_color};" href="javascript:${functionCall}">${word}</a> `;
};

const wrapWordForIndividualPlayback = (word, word_idx) => {
  return `<a onmouseover="generateWordModal(${word_idx})" style="white-space:nowrap;" href="javascript:playNativeAndRecordedWord(${word_idx})">${word}</a> `;
};
