package org.pytorch.demo.vision;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.SystemClock;
import android.text.TextUtils;
import android.util.Log;
import android.view.TextureView;
import android.view.View;
import android.view.ViewStub;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.demo.Constants;
import org.pytorch.demo.R;
import org.pytorch.demo.Utils;
import org.pytorch.demo.vision.view.ResultRowView;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.LinkedList;
import java.util.Locale;
import java.util.Queue;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;

public class ImageClassificationActivity extends AbstractCameraXActivity<ImageClassificationActivity.AnalysisResult> {

  public static final String INTENT_MODULE_ASSET_NAME = "INTENT_MODULE_ASSET_NAME";
  public static final String INTENT_INFO_VIEW_TYPE = "INTENT_INFO_VIEW_TYPE";

  private static final int INPUT_TENSOR_WIDTH = 224;
  private static final int INPUT_TENSOR_HEIGHT = 224;
  private static final int TOP_K = 3;
  private static final int MOVING_AVG_PERIOD = 10;
  private static final String FORMAT_MS = "%dms";
  private static final String FORMAT_AVG_MS = "avg:%.0fms";

  private static final String FORMAT_FPS = "%.1fFPS";
  public static final String SCORES_FORMAT = "%.2f";

  static class AnalysisResult {

    private final Bitmap bitmap;

    private final String[] topNClassNames;
    private final float[] topNScores;
    private final long analysisDuration;
    private final long moduleForwardDuration;

    public AnalysisResult(Bitmap bitmap, String[] topNClassNames, float[] topNScores,
                          long moduleForwardDuration, long analysisDuration) {
      this.bitmap = bitmap;

      this.topNClassNames = topNClassNames;
      this.topNScores = topNScores;
      this.moduleForwardDuration = moduleForwardDuration;
      this.analysisDuration = analysisDuration;
    }
  }

  private boolean mAnalyzeImageErrorState;
  private ResultRowView[] mResultRowViews = new ResultRowView[TOP_K];
  private ImageView mImageView;
  private TextView mFpsText;
  private TextView mMsText;
  private TextView mMsAvgText;
  private Module mModule;
  private String mModuleAssetName;
  private FloatBuffer mInputTensorBuffer;
  private Tensor mInputTensor;
  private long mMovingAvgSum = 0;
  private Queue<Long> mMovingAvgQueue = new LinkedList<>();

  @Override
  protected int getContentViewLayoutId() {
    return R.layout.activity_image_classification;
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    final ResultRowView headerResultRowView =
        findViewById(R.id.image_classification_result_header_row);
    headerResultRowView.nameTextView.setText(R.string.image_classification_results_header_row_name);
    headerResultRowView.scoreTextView.setText(R.string.image_classification_results_header_row_score);

    mResultRowViews[0] = findViewById(R.id.image_classification_top1_result_row);
    mResultRowViews[1] = findViewById(R.id.image_classification_top2_result_row);
    mResultRowViews[2] = findViewById(R.id.image_classification_top3_result_row);

    mImageView = findViewById(R.id.image_view);
    mFpsText = findViewById(R.id.image_classification_fps_text);
    mMsText = findViewById(R.id.image_classification_ms_text);
    mMsAvgText = findViewById(R.id.image_classification_ms_avg_text);
  }

  @Override
  protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
    mMovingAvgSum += result.moduleForwardDuration;
    mMovingAvgQueue.add(result.moduleForwardDuration);
    if (mMovingAvgQueue.size() > MOVING_AVG_PERIOD) {
      mMovingAvgSum -= mMovingAvgQueue.remove();
    }

    mImageView.setImageBitmap(result.bitmap);

    for (int i = 0; i < TOP_K; i++) {
      final ResultRowView rowView = mResultRowViews[i];
      rowView.nameTextView.setText(result.topNClassNames[i]);
      rowView.scoreTextView.setText(String.format(Locale.US, SCORES_FORMAT,
          result.topNScores[i]));
      rowView.setProgressState(false);
    }

    mMsText.setText(String.format(Locale.US, FORMAT_MS, result.moduleForwardDuration));
    if (mMsText.getVisibility() != View.VISIBLE) {
      mMsText.setVisibility(View.VISIBLE);
    }
    mFpsText.setText(String.format(Locale.US, FORMAT_FPS, (1000.f / result.analysisDuration)));
    if (mFpsText.getVisibility() != View.VISIBLE) {
      mFpsText.setVisibility(View.VISIBLE);
    }

    if (mMovingAvgQueue.size() == MOVING_AVG_PERIOD) {
      float avgMs = (float) mMovingAvgSum / MOVING_AVG_PERIOD;
      mMsAvgText.setText(String.format(Locale.US, FORMAT_AVG_MS, avgMs));
      if (mMsAvgText.getVisibility() != View.VISIBLE) {
        mMsAvgText.setVisibility(View.VISIBLE);
      }
    }
  }

  protected String getModuleAssetName() {
    if (!TextUtils.isEmpty(mModuleAssetName)) {
      return mModuleAssetName;
    }
    final String moduleAssetNameFromIntent = getIntent().getStringExtra(INTENT_MODULE_ASSET_NAME);
    mModuleAssetName = !TextUtils.isEmpty(moduleAssetNameFromIntent)
        ? moduleAssetNameFromIntent
        : "resnet18.pt";

    return mModuleAssetName;
  }

  @Override
  protected String getInfoViewAdditionalText() {
    return "\nModel file name: " + getModuleAssetName();
  }

  @Override
  @WorkerThread
  @Nullable
  protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
    if (mAnalyzeImageErrorState) {
      return null;
    }

    try {
      if (mModule == null) {
        final String moduleFileAbsoluteFilePath = new File(
            Utils.assetFilePath(this, getModuleAssetName())).getAbsolutePath();
        Log.d("debug", getModuleAssetName());
        Log.d("debug", moduleFileAbsoluteFilePath);
        mModule = Module.load(moduleFileAbsoluteFilePath);

        mInputTensorBuffer =
            Tensor.allocateFloatBuffer(3 * INPUT_TENSOR_WIDTH * INPUT_TENSOR_HEIGHT);
        mInputTensor = Tensor.fromBlob(mInputTensorBuffer, new long[]{1, 3, INPUT_TENSOR_HEIGHT, INPUT_TENSOR_WIDTH});
      }

      final long startTime = SystemClock.elapsedRealtime();
      TensorImageUtils.imageYUV420CenterCropToFloatBuffer(
          image.getImage(), rotationDegrees,
          INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT,
          new float[] { 0.5f, 0.5f, 0.5f },
          new float[] { 0.5f, 0.5f, 0.5f },
          mInputTensorBuffer, 0);

      final long moduleForwardStartTime = SystemClock.elapsedRealtime();

      final FloatBuffer InputTensorBuffer =
              Tensor.allocateFloatBuffer(3 * 256 * 256);

      for (int i = 0; i < 224; i++) {
        for (int j = 0; j < 224; j++) {
          InputTensorBuffer.put((j + 16) * 256 + i + 16, mInputTensorBuffer.get(j * 224 + i));
        }
      }

      final Tensor InputTensor = Tensor.fromBlob(InputTensorBuffer, new long[]{1, 3, 256, 256});

      final Tensor outputTensor = mModule.forward(IValue.from(mInputTensor)).toTensor();
      float[] outputArray = outputTensor.getDataAsFloatArray();

      Bitmap bitmap = Bitmap.createBitmap(INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT, Bitmap.Config.ARGB_8888);
      final int size = INPUT_TENSOR_WIDTH * INPUT_TENSOR_HEIGHT;
      for (int i = 0; i < size; i++) {
        final int x = i % INPUT_TENSOR_WIDTH;
        final int y = i / INPUT_TENSOR_HEIGHT;
//        final int b = (int) ((mInputTensorBuffer.get(i) * 0.5 + 0.5) * 255);
//        final int g = (int) ((mInputTensorBuffer.get(size + i) * 0.5 + 0.5) * 255);
//        final int r = (int) ((mInputTensorBuffer.get(size * 2 + i) * 0.5 + 0.5) * 255);
        final int r = (int) ((outputArray[i] * 0.5 + 0.5) * 255);
        final int g = (int) ((outputArray[size + i] * 0.5 + 0.5) * 255);
        final int b = (int) ((outputArray[size * 2 + i] * 0.5 + 0.5) * 255);
        final int c = (r + g + b) / 3;
        bitmap.setPixel(x, y, 0xFF000000 | (c << 16) | (c << 8) | c);
      }

      final long moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime;

      final long analysisDuration = SystemClock.elapsedRealtime() - startTime;
      return new AnalysisResult(bitmap, new String[] { "a", "b", "c" }, new float[] { 1.0f, 1.0f, 1.0f }, moduleForwardDuration, analysisDuration);
    } catch (Exception e) {
      Log.e(Constants.TAG, "Error during image analysis", e);
      mAnalyzeImageErrorState = true;
      runOnUiThread(() -> {
        if (!isFinishing()) {
          showErrorDialog(v -> ImageClassificationActivity.this.finish());
        }
      });
      return null;
    }
  }

  @Override
  protected int getInfoViewCode() {
    return getIntent().getIntExtra(INTENT_INFO_VIEW_TYPE, -1);
  }

  @Override
  protected void onDestroy() {
    super.onDestroy();
    if (mModule != null) {
      mModule.destroy();
    }
  }
}
