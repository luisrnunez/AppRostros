package com.example.appreconocimientofacial;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraManager;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Size;
import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import com.example.appreconocimientofacial.ml.ModelUnquant;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.mlkit.vision.text.Text;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainActivity extends AppCompatActivity
        implements
        ImageReader.OnImageAvailableListener{


    public static int REQUEST_CAMERA = 111;
    public static int REQUEST_GALLERY = 222;

    public Bitmap mSelectedImage;
    public ImageView mImageView;
    public TextView txtResults;
    ArrayList<String> permisosNoAprobados;
    public Button btCamera, btGaleria;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        txtResults = findViewById(R.id.txtresults);
        btCamera = findViewById(R.id.btCamera);
        btGaleria = findViewById(R.id.btGallery);

        ArrayList<String> permisos_requeridos = new ArrayList<String>();
        permisos_requeridos.add(Manifest.permission.CAMERA);
        permisos_requeridos.add(Manifest.permission.MANAGE_EXTERNAL_STORAGE);
        permisos_requeridos.add(Manifest.permission.READ_EXTERNAL_STORAGE);

        permisosNoAprobados  = getPermisosNoAprobados(permisos_requeridos);

        requestPermissions(permisosNoAprobados.toArray(new String[permisosNoAprobados.size()]),
                111);
    }
    public void abrirCamera (View view){
        setFragment();
    }

    public void abrirGaleria (View view){

        Intent i = new Intent(Intent.ACTION_PICK,
                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(i, REQUEST_GALLERY);
    }
    public ArrayList<String> getPermisosNoAprobados(ArrayList<String>  listaPermisos) {
        ArrayList<String> list = new ArrayList<String>();
        Boolean habilitado;
        if (Build.VERSION.SDK_INT >= 23)
            for(String permiso: listaPermisos) {
                if (checkSelfPermission(permiso) != PackageManager.PERMISSION_GRANTED) {
                    list.add(permiso);
                    habilitado = false;
                }else
                    habilitado=true;

                if(permiso.equals(Manifest.permission.CAMERA))
                    btCamera.setEnabled(habilitado);
                else if (permiso.equals(Manifest.permission.MANAGE_EXTERNAL_STORAGE)  ||
                        permiso.equals(Manifest.permission.READ_EXTERNAL_STORAGE))
                    btGaleria.setEnabled(habilitado);
            }
        return list;
    }
     @Override
     public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
         super.onRequestPermissionsResult(requestCode, permissions, grantResults);

         for(int i=0; i<permissions.length; i++){
             if(permissions[i].equals(Manifest.permission.CAMERA)){
                 btCamera.setEnabled(grantResults[i] == PackageManager.PERMISSION_GRANTED);
             } else if(permissions[i].equals(Manifest.permission.MANAGE_EXTERNAL_STORAGE) ||
                     permissions[i].equals(Manifest.permission.READ_EXTERNAL_STORAGE)
             ) {
                 btGaleria.setEnabled(grantResults[i] == PackageManager.PERMISSION_GRANTED);
             }
         }
     }

     int previewHeight = 0,previewWidth = 0;
     int sensorOrientation;
     @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
     protected void setFragment() {
         final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
         String cameraId = null;
         try {
             cameraId = manager.getCameraIdList()[0];
         } catch (CameraAccessException e) {
             e.printStackTrace();
         }
         CameraConnectionFragment fragment;
         CameraConnectionFragment camera2Fragment =
                 CameraConnectionFragment.newInstance(
                         new CameraConnectionFragment.ConnectionCallback() {
                             @Override
                             public void onPreviewSizeChosen(final Size size, final int rotation) {
                                 previewHeight = size.getHeight(); previewWidth = size.getWidth();
                                 sensorOrientation = rotation - getScreenOrientation();
                             }
                         },
                         this, R.layout.camera_fragment, new Size(640, 480));
         camera2Fragment.setCamera(cameraId);
         fragment = camera2Fragment;
         getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit();
     }

     protected int getScreenOrientation() {
         switch (getWindowManager().getDefaultDisplay().getRotation()) {
             case Surface.ROTATION_270:
                 return 270;
             case Surface.ROTATION_180:
                 return 180;
             case Surface.ROTATION_90:
                 return 90;
             default:
                 return 0;
         }
     }
    private boolean isProcessingFrame = false;
    private byte[][] yuvBytes = new byte[3][];
    private int[] rgbBytes = null;
    private int yRowStride;
    private Runnable postInferenceCallback;
    private Runnable imageConverter;
    private Bitmap rgbFrameBitmap;
    private Bitmap bitmap;
    @Override
    public void onImageAvailable(ImageReader imageReader) {
        if (previewWidth == 0 || previewHeight == 0) return;
        if (rgbBytes == null) rgbBytes = new int[previewWidth * previewHeight];
        try {
            final Image image = imageReader.acquireLatestImage();
            if (image == null) return;
            if (isProcessingFrame) {
                image.close();
                return;
            }
            isProcessingFrame = true;
            final Image.Plane[] planes = image.getPlanes();
            fillBytes(planes, yuvBytes);
            yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();
            imageConverter = new Runnable() {
                @Override
                public void run() {
                    ImageUtils.convertYUV420ToARGB8888(yuvBytes[0], yuvBytes[1], yuvBytes[2], previewWidth, previewHeight,
                            yRowStride, uvRowStride, uvPixelStride, rgbBytes);

                    // Convert the ARGB byte array to a Bitmap
                    bitmap = Bitmap.createBitmap(rgbBytes, previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
                    // Process the Bitmap
                }
            };
            postInferenceCallback = new Runnable() {
                @Override
                public void run() {
                    image.close();
                    isProcessingFrame = false;
                }
            };

            processImage();

        } catch (final Exception e) {
            // Handle exceptions
        }
    }

    protected void fillBytes(final Image.Plane[] planes, final byte[][] yuvBytes) {
        for (int i = 0; i < planes.length; ++i) {
            final ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null) {
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBytes[i]);
        }
    }

    String[] labels = {"ROSTRO", "ROSTRONO"};
    int cont = 0;
    private void processImage() {
        imageConverter.run();

        try {
            ModelUnquant model = ModelUnquant.newInstance(getApplicationContext());
            //txtResults.setText("1");
            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
            // inputFeature0.loadBuffer(TensorImage.fromBitmap(bitmap).getBuffer());
            //txtResults.setText("2");
            // Runs model inference and gets result.

            //txtResults.setText("3");
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
            byteBuffer.order(ByteOrder.nativeOrder());
            //txtResults.setText("4");
            // pixeles de imagen
            int [] intValues = new int[224 * 224];
            bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

            // normalizar la imagen en el bytebuffer.
            int pixel = 0;
            for(int i = 0; i < 224; i++){
                for(int j = 0; j < 224; j++){
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }
            //txtResults.setText("6");
            inputFeature0.loadBuffer(byteBuffer);
            ModelUnquant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] probabilities = outputFeature0.getFloatArray();

// Obtener la probabilidad de "Rostro" (índice 0) y "No_Rostro" (índice 1)
            float probRostro = probabilities[0];
            float probNoRostro = probabilities[1];

            String resultMessage;

            if (probRostro > 0.5) {
                resultMessage = "Es un rostro:   " + probRostro;
                if (probRostro > 0.9f) {
                    // Mostrar un Toast si la probabilidad es mayor a 0.9
                    Toast.makeText(this, "ES UN ROSTRO", Toast.LENGTH_SHORT).show();
                }

            } else {
                resultMessage = "No es un rostro:    " + probRostro;
            }

            txtResults.setText(resultMessage);

        } catch (IOException e) {
            txtResults.setText("Error al procesar Modelo");
        }

        postInferenceCallback.run();
    }

    int getMax(float[] arr) {
        int max = 0;
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] > arr[max]) max = i;
        }
        return max;
    }


}

class CategoryComparator implements java.util.Comparator<Category> {
    @Override
    public int compare(Category a, Category b) {
        return (int)(b.getScore()*100) - (int)(a.getScore()*100);
    }
}

