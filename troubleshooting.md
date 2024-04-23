```
#Jeżeli po instalacji cudnn wyskakuje błąd typu: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
#https://stackoverflow.com/questions/53698035/failed-to-get-convolution-algorithm-this-is-probably-because-cudnn-failed-to-in
#To albo nie masz wystarczającej ilości pamięci GPU lub masz złą wersję cudnn dla cudatoolkit
#Możesz dodać do kodu poniższe dwa polecenia i może pomogą z pamięcią:
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

#Z tymi dwoma linijkami kodu zadziałało- czyli jest to raczej wina braku pamięci (https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth)
#po prostu tensorflow alokuje całą pamięć GPU aby zwiększyć wydajność i brakuje pamięci dla pytorcha. Pytorch tak nie działa on alokuje pamięci tyle ile w danej chwili potrzebuje:
#W pliku homography.py ustawić memory_limit=1024:
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

#Działa z wersjami ze różnych źródeł condy i archiwów nvidii:
#Wersje 7.6.0, 7.6.5 pobrane z https://developer.nvidia.com/rdp/cudnn-archive i wrzucone do ~/miniconda3/envs/dt2hom2GPU2/lib też działają
conda install -c pytorch cudnn=7.6.5
conda install -c conda-forge cudnn
conda install -c anaconda cudnn
```
```
#### ModuleNotFoundError: No module named '_sysconfigdata_x86_64_conda_linux_gnu'

conda install gcc_linux-64 gxx_linux-64 -y
ln -s ~/miniconda3/envs/dt2hom2GPU/bin/x86_64-conda-linux-gnu-gcc ~/miniconda3/envs/dt2hom2GPU/bin/gcc
ln -s ~/miniconda3/envs/dt2hom2GPU/bin/x86_64-conda-linux-gnu-g++ ~/miniconda3/envs/dt2hom2GPU/bin/g++
cp ~/miniconda3/envs/dt2hom2GPU/lib/python3.7/_sysconfigdata_x86_64_conda_cos6_linux_gnu.py ~/miniconda3/envs/dt2hom2GPU/lib/python3.7/_sysconfigdata_x86_64_conda_linux_gnu.py
```
```
./common/maskApi.c:8:10: fatal error: math.h: Nie ma takiego pliku ani katalogu
      8 | #include <math.h>
        |          ^~~~~~~~
  compilation terminated.
  error: command '/usr/bin/gcc' failed with exit code 1
  ----------------------------------------
  ERROR: Failed building wheel for pycocotools
  
conda install gcc_linux-64 gxx_linux-64 -y
ln -s ~/miniconda3/envs/TV2D/bin/x86_64-conda-linux-gnu-gcc ~/miniconda3/envs/TV2D/bin/gcc
ln -s ~/miniconda3/envs/TV2D/bin/x86_64-conda-linux-gnu-g++ ~/miniconda3/envs/TV2D/bin/g++
cp ~/miniconda3/envs/TV2D/lib/python3.7/_sysconfigdata_x86_64_conda_cos6_linux_gnu.py ~/miniconda3/envs/TV2D/lib/python3.8/_sysconfigdata_x86_64_conda_linux_gnu.py
```

```
  /home/jan/miniconda3/envs/detectron2/compiler_compat/ld: cannot find /lib64/libpthread.so.0
  /home/jan/miniconda3/envs/detectron2/compiler_compat/ld: cannot find /usr/lib64/libpthread_nonshared.a
  collect2: error: ld returned 1 exit status
  error: command '/home/jan/miniconda3/envs/detectron2/bin/gcc' failed with exit code 1
  ----------------------------------------
  ERROR: Failed building wheel for pycocotools
Failed to build pycocotools
ERROR: Could not build wheels for pycocotools which use PEP 517 and cannot be installed directly
##Instalowanie libpthread-stubs0-dev nic nie dało
sudo apt install libpthread-stubs0-dev
##Miałem zaisntalowany google chrome remote destkop może to to ale raczej nie... sudo apt-get remove chrome-remote-desktop
##Zainstalowałem jeszcze python3-dev ale to chyba też nic nie zmieniło
##Istalowałem potem wdł poradnika dla samego detectrona2 czyli na pythonie=3.8 
conda create --name detectron2 python=3.8
conda activate detectron2
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
conda install -c conda-forge opencv -y
conda install -c anaconda git -y
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
#i to zadziałało i potem już działało ale nie wiem cz to to...
```
```
###
find ninja.. Falling back to using the slow distutils backend.
      warnings.warn(msg.format('we could not find ninja.'))
conda install -c conda-forge ninja
```

```
###
    subprocess.CalledProcessError: Command '['which', 'g++']' returned non-zero exit status 1.
    ----------------------------------------
ERROR: Command errored out with exit status 1: /home/jan/miniconda3/envs/dt2raw/bin/python -c 'import io, os, sys, setuptools, tokenize; f = getattr(tokenize, '"'"'open'"'"', open)(__file__) if os.path.exists(__file__) else io.StringIO('"'"'from setuptools import setup; setup()'"'"');code = f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' develop --no-deps Check the logs for full command output.
conda install gcc_linux-64 gxx_linux-64 -y
ln -s ~/miniconda3/envs/dt2raw/bin/x86_64-conda-linux-gnu-gcc ~/miniconda3/envs/dt2raw/bin/gcc
ln -s ~/miniconda3/envs/dt2raw/bin/x86_64-conda-linux-gnu-g++ ~/miniconda3/envs/dt2raw/bin/g++
```
```
#### Could not load dynamic library 'libcudnn.so.8';
2022-03-15 13:49:47.659855: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudnn.so.8'; dlerror: libcudnn.so.8: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /media/jan/Dane2/miniconda3envs/nbiml/lib/python3.8/site-packages/cv2/../../lib64:
#####Jednorazowa naprawa:
export LD_LIBRARY_PATH=/media/jan/Dane2/miniconda3envs/nbiml/lib/
export LD_LIBRARY_PATH=/home/jan/miniconda3/envs/dt2homDeep3060/lib
#####Stała naprawa:
conda env config vars set LD_LIBRARY_PATH=/home/jan/miniconda3/envs/dt2homDeep3060/lib
```

```
#### AttributeError: module 'keras.utils' has no attribute 'get_file'
Traceback (most recent call last):
  File "main.py", line 187, in <module>
    homographyDetector = homography.Homography(threshold=0.9)
  File "/home/jan/Dokumenty/UAM/2stopien/sem2/Seminarium/TV2D/homography.py", line 28, in __init__
    self.homo_estimator = HomographyEstimator()
  File "/home/jan/Dokumenty/UAM/2stopien/sem2/Seminarium/TV2D/narya/narya/tracker/homography_estimator.py", line 50, in __init__
    backbone="efficientnetb3", num_classes=29, input_shape=keypoint_model_input_shape,
  File "/home/jan/Dokumenty/UAM/2stopien/sem2/Seminarium/TV2D/narya/narya/models/keras_models.py", line 125, in __init__
    encoder_weights="imagenet",
  File "/home/jan/miniconda3/envs/dt2homDeep3060/lib/python3.7/site-packages/segmentation_models/__init__.py", line 34, in wrapper
    return func(*args, **kwargs)
  File "/home/jan/miniconda3/envs/dt2homDeep3060/lib/python3.7/site-packages/segmentation_models/models/fpn.py", line 227, in FPN
    **kwargs,
  File "/home/jan/miniconda3/envs/dt2homDeep3060/lib/python3.7/site-packages/segmentation_models/backbones/backbones_factory.py", line 103, in get_backbone
    model = model_fn(*args, **kwargs)
  File "/home/jan/miniconda3/envs/dt2homDeep3060/lib/python3.7/site-packages/classification_models/models_factory.py", line 78, in wrapper
    return func(*args, **new_kwargs)
  File "/home/jan/miniconda3/envs/dt2homDeep3060/lib/python3.7/site-packages/efficientnet/model.py", line 538, in EfficientNetB3
    **kwargs)
  File "/home/jan/miniconda3/envs/dt2homDeep3060/lib/python3.7/site-packages/efficientnet/model.py", line 470, in EfficientNet
    weights_path = keras_utils.get_file(file_name,
AttributeError: module 'keras.utils' has no attribute 'get_file'
### Rozwiązanie (tam gdzie jest importowanie segmentation_models czyli w pliku /home/jan/Dokumenty/UAM/2stopien/sem2/Seminarium/TV2D/narya/narya/models/keras_models.py) :
import segmentation_models as sm

sm.set_framework('tf.keras')

sm.framework()

https://stackoverflow.com/questions/67792138/attributeerror-module-keras-utils-has-no-attribute-get-file-using-segmentat
```

```
Traceback (most recent call last):
  File "main.py", line 316, in <module>
    cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
cv2.error: OpenCV(4.5.5) /io/opencv/modules/highgui/src/window.cpp:1251: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvNamedWindow'

pip uninstall opencv-python
pip install opencv-python
https://stackoverflow.com/questions/65019604/attributeerror-module-cv2-has-no-attribute-version
X->https://stackoverflow.com/questions/67120450/error-2unspecified-error-the-function-is-not-implemented-rebuild-the-libra
```

```
# Przy treonowaniu keypoints...
119/127 [===========================>..] - ETA: 2s - loss: 0.9839 - iou_score: 0.3797 - f1-score: 0.38222022-04-03 00:14:44.720184: W tensorflow/core/framework/op_kernel.cc:1733] UNKNOWN: error: OpenCV(4.5.5) /io/opencv/modules/imgproc/src/color.cpp:182: error: (-215:Assertion failed) !_src.empty() in function 'cvtColor'

Usunąłem pliki .DS_Store z wszytkich katalogów datasetu i to pomogło
```

```
#### AttributeError: module 'keras.utils' has no attribute 'Sequence'
keras.utils.all_utils.Sequence
```

```
#### AttributeError: module 'keras.optimizers' has no attribute 'Adam'
optim = tf.keras.optimizers.Adam(opt.lr)
```

```
#### Generowanie pliku pythonowego z pliku ui (Qt Designer)
pyside2-uic form.ui -o form_ui.py
https://www.badprog.com/python-3-pyside2-setting-up-and-using-qt-designer
```

```
#### Budowanie narzędzia do wizualizacji:
conda activate TV2D
cd vizualize_tool/visualize/
pyinstaller -F --add-data "form_ui.py:." --add-data "resources/world_cup_template.png:resources/world_cup_template.png" --add-data "detected_object.py:." widget.py
#### W Windowsie zamiast : należy użyć ;
pyinstaller -F --add-data "form_ui.py;." --add-data "resources/world_cup_template.png;resources/world_cup_template.png" --add-data "detected_object.py;." widget.py
#### Budowanie nie działa do końca prawidłowow...
```

```
Traceback (most recent call last):
  File "demo/demo.py", line 176, in <module>
    for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
  File "/home/jan/miniconda3/envs/dt2homDeep3060/lib/python3.7/site-packages/tqdm/std.py", line 1195, in __iter__
    for obj in iterable:
  File "/home/jan/Dokumenty/Chwila/detectron2/demo/predictor.py", line 129, in run_on_video
    yield process_predictions(frame, self.predictor(frame))
  File "/home/jan/miniconda3/envs/dt2homDeep3060/lib/python3.7/site-packages/detectron2/engine/defaults.py", line 317, in __call__
    predictions = self.model([inputs])[0]
  File "/home/jan/miniconda3/envs/dt2homDeep3060/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/jan/miniconda3/envs/dt2homDeep3060/lib/python3.7/site-packages/detectron2/modeling/meta_arch/rcnn.py", line 146, in forward
    return self.inference(batched_inputs)
  File "/home/jan/miniconda3/envs/dt2homDeep3060/lib/python3.7/site-packages/detectron2/modeling/meta_arch/rcnn.py", line 206, in inference
    assert "proposals" in batched_inputs[0]
AssertionError
#### Użyty został model który wymaga dostarczenia propozycji regionów np. problem dla Fast-RCNN
https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-input-format
https://stackoverflow.com/questions/71443140/what-is-the-difference-between-the-faster-r-cnn-and-rpn-fast-r-cnn-models-offe
```