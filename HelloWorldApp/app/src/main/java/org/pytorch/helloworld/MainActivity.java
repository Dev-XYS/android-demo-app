package org.pytorch.helloworld;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;

import androidx.annotation.NonNull;
import androidx.annotation.WorkerThread;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class MainActivity extends AppCompatActivity {

  private static final int REQUEST_CODE_PERMISSIONS = 10;
  private static final String[] REQUIRED_PERMISSIONS = { Manifest.permission.CAMERA };

  private ImageView mImageView;
  private TextView mTextView;

  private static final int INPUT_TENSOR_WIDTH = 256;
  private static final int INPUT_TENSOR_HEIGHT = 256;

  private FloatBuffer mInputTensorBuffer;
  private Tensor mInputTensor;
  private float[] mInputTensorArray;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    mImageView = findViewById(R.id.image);
    mTextView = findViewById(R.id.text);

    // Request camera permissions
    if (allPermissionsGranted()) {
      startCamera();
    } else {
      ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
    }
  }

  @WorkerThread
  private void analyzeImage(ImageProxy image, int rotationDegrees) {
    int size = INPUT_TENSOR_WIDTH * INPUT_TENSOR_HEIGHT;

    mInputTensorBuffer =
            Tensor.allocateFloatBuffer(3 * INPUT_TENSOR_WIDTH * INPUT_TENSOR_HEIGHT);
    mInputTensor = Tensor.fromBlob(mInputTensorBuffer, new long[] {1, 3, INPUT_TENSOR_HEIGHT, INPUT_TENSOR_WIDTH});

    TensorImageUtils.imageYUV420CenterCropToFloatBuffer(
            image.getImage(), rotationDegrees,
            INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB,
            mInputTensorBuffer, 0);

    float[] array = mInputTensor.getDataAsFloatArray();

    ByteBuffer buffer = ByteBuffer.allocate(4 * size);

    for (int i = 0; i < size; i++) {
//      buffer.put(i * 4, (byte) (mInputTensorArray[i] * 128 + 128));
//      buffer.put(i * 4 + 1, (byte) (mInputTensorArray[size + i] * 128 + 128));
//      buffer.put(i * 4 + 2, (byte) (mInputTensorArray[size * 2 + i] * 128 + 128));
//      buffer.put(i * 4 + 3, (byte) 255);
      buffer.put(i * 4, (byte) (array[i] * 128 + 128));
      buffer.put(i * 4 + 1, (byte) (array[size + i] * 128 + 128));
      buffer.put(i * 4 + 2, (byte) (array[size * 2 + i] * 128 + 128));
      buffer.put(i * 4 + 3, (byte) 255);
    }

    buffer.rewind();

    Bitmap bitmap = Bitmap.createBitmap(INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT, Bitmap.Config.ARGB_8888);
    bitmap.copyPixelsFromBuffer(buffer);
    mImageView.setImageBitmap(bitmap);
  }

  private void startCamera() {
    final TextureView textureView = getCameraPreviewTextureView();
    final PreviewConfig previewConfig = new PreviewConfig.Builder().build();
    final Preview preview = new Preview(previewConfig);

    final ImageAnalysisConfig imageAnalysisConfig =
            new ImageAnalysisConfig.Builder()
                    .setTargetResolution(new Size(INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT))
                    .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
                    .build();
    final ImageAnalysis imageAnalysis = new ImageAnalysis(imageAnalysisConfig);
    imageAnalysis.setAnalyzer(this::analyzeImage);
    mInputTensorArray = new float[3 * INPUT_TENSOR_WIDTH * INPUT_TENSOR_HEIGHT];

    CameraX.bindToLifecycle(this, preview, imageAnalysis);
  }

  /**
   * Process result from permission request dialog box, has the request
   * been granted? If yes, start Camera. Otherwise display a toast
   */
  @Override
  public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
    // super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    if (requestCode == REQUEST_CODE_PERMISSIONS) {
      if (allPermissionsGranted()) {
        startCamera();
      } else {
        Toast.makeText(this,
                "Permissions not granted by the user.",
                Toast.LENGTH_SHORT).show();
        finish();
      }
    }
  }

  /**
   * Check if all permission specified in the manifest have been granted
   */
  private boolean allPermissionsGranted() {
    for (String permission : REQUIRED_PERMISSIONS) {
      if (ContextCompat.checkSelfPermission(getBaseContext(), permission) != PackageManager.PERMISSION_GRANTED) {
        return false;
      }
    }
    return true;
  }
}
