package com.pes.mldemo

import android.os.Bundle

import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.google.mlkit.genai.common.DownloadCallback
import com.google.mlkit.genai.common.FeatureStatus
import com.google.mlkit.genai.common.GenAiException
import com.google.mlkit.genai.summarization.Summarization
import com.google.mlkit.genai.summarization.SummarizationRequest
import com.google.mlkit.genai.summarization.Summarizer
import com.google.mlkit.genai.summarization.SummarizerOptions
import com.pes.mldemo.ui.theme.MLDemoTheme
import kotlinx.coroutines.guava.await
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    private var summarizer: Summarizer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val options = SummarizerOptions.builder(this)
            .setInputType(SummarizerOptions.InputType.ARTICLE)
            .setOutputType(SummarizerOptions.OutputType.THREE_BULLETS)
            .setLanguage(SummarizerOptions.Language.ENGLISH)
            .build()
        summarizer = Summarization.getClient(options)

        enableEdgeToEdge()
        setContent {
            MLDemoTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    SummarizationScreen(modifier = Modifier.padding(innerPadding)) { article, onStatus, onResult ->
                        lifecycleScope.launch {
                            runSummarization(article, onStatus, onResult)
                        }

                    }
                }
            }
        }
    }

    private suspend fun runSummarization(
        article: String,
        onStatus: (String) -> Unit,
        onResult: (String) -> Unit
    ) {
        val client = summarizer ?: return
        try {
            onStatus("Checking on-device GenAI availability…")
            when (client.checkFeatureStatus().await()) {
                FeatureStatus.UNAVAILABLE -> {
                    onResult("Summarization is unavailable on this device. Test on a supported physical device with Android AICore and Gemini Nano.")
                    return
                }
                FeatureStatus.DOWNLOADABLE -> {
                    onStatus("Downloading the required on-device model…")
                    client.downloadFeature(object : DownloadCallback {
                        override fun onDownloadStarted(bytesToDownload: Long) {
                            onStatus("Model download started…")
                        }
                        override fun onDownloadProgress(totalBytesDownloaded: Long) {
                            onStatus("Downloading model: $totalBytesDownloaded bytes received")
                        }
                        override fun onDownloadCompleted() {
                            onStatus("Model download complete.")
                        }
                        override fun onDownloadFailed(e: GenAiException) {
                            onStatus("Model download failed: ${e.message}")
                        }
                    }).await()
                }
                FeatureStatus.DOWNLOADING -> onStatus("Model is downloading; waiting for inference…")
                FeatureStatus.AVAILABLE -> onStatus("Model is available. Generating summary…")
            }

            val request = SummarizationRequest.builder(article).build()
            onStatus("Generating summary on this device…")
            val result = client.runInference(request).await()
            onResult(result.summary)
            onStatus("Completed")
        } catch (e: Exception) {
            onResult("Summarization failed: ${e.message ?: e.javaClass.simpleName}")
            onStatus("Error")
        }
    }

    override fun onDestroy() {
        summarizer?.close()
        summarizer = null
        super.onDestroy()
    }
}

@Composable
private fun SummarizationScreen(modifier: Modifier,
    onSummarize: (String, (String) -> Unit, (String) -> Unit) -> Unit
) {
    val sampleArticle = "Machine learning is a branch of artificial intelligence that enables computer systems to learn patterns from data and make predictions without being explicitly programmed for every possible situation. Instead of following only fixed rules, a machine learning model is trained using examples. During training, the model adjusts its internal parameters to reduce the difference between its predictions and expected results. Once trained, the model can process new data and produce predictions or classifications. Machine learning is used in everyday applications, including spam filtering, recommendation systems, speech recognition, image classification, fraud detection, and medical image analysis. The quality of a model depends on the amount and quality of training data, the choice of algorithm, and how carefully the model is evaluated. Developers must also consider privacy, fairness, accuracy, and the possibility of incorrect predictions when deploying machine learning systems."

    var article by remember { mutableStateOf(sampleArticle) }
    var status by remember { mutableStateOf("Ready. Enter an article over 400 characters.") }
    var summary by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }

    Column(
        modifier = modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text("ML Kit GenAI", style = MaterialTheme.typography.headlineMedium)
        Text("On-device Summarization API", style = MaterialTheme.typography.titleLarge)
        Text("Summarizes an article into three bullet points using Gemini Nano through Android AICore.")

        OutlinedTextField(
            value = article,
            onValueChange = { article = it },
            modifier = Modifier.fillMaxWidth().height(250.dp),
            label = { Text("Article text") },
            supportingText = { Text("${article.length} characters • ARTICLE input must exceed 400 characters") }
        )

        Button(
            onClick = {
                summary = ""
                busy = true
                onSummarize(
                    article,
                    { status = it },
                    { summary = it; busy = false }
                )
            },
            enabled = !busy && article.length > 400,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(if (busy) "Working…" else "Summarize")
        }

        if (busy) {
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                CircularProgressIndicator()
                Text(status)
            }
        } else {
            Text("Status: $status")
        }

        if (summary.isNotBlank()) {
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp)) {
                    Text("Generated summary", style = MaterialTheme.typography.titleMedium)
                    Spacer(Modifier.height(8.dp))
                    Text(summary)
                }
            }
        }

        Text(
            "Device note: the API is not supported on every Android phone or emulator. Check the runtime status shown above.",
            style = MaterialTheme.typography.bodySmall
        )
    }
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    MLDemoTheme {
        Greeting("Android")
    }
}