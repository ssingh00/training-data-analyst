	�u�X��?�u�X��?!�u�X��?	������"@������"@!������"@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�u�X��?�~�n��?A�s�ڝ�?Y�E&��H�?*	��K7���@2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat::Prefetch::Map::BatchV2::ShuffleAndRepeat::LegacyParallelInterleaveV2 zS�
c�?!��� I@)zS�
c�?1��� I@:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat::Prefetch::Map::BatchV2::ShuffleAndRepeat::LegacyParallelInterleaveV2[0]::CSV V�6���?!�	���/@)V�6���?1�	���/@:Preprocessing2�
ZIterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat::Prefetch::Map::BatchV2���2���?!���U>4Q@)�G�C���?1�z �m#@:Preprocessing2�
LIterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat::PrefetchHO�C���?!��}Q#@)HO�C���?1��}Q#@:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat::Prefetch::Map::BatchV2::ShuffleAndRepeat ���a�<�?!H?5�M@)ڬ�\m��?1��{�;2"@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism5c�tv2�?!.�nȒ@)�(z�c��?1�h��F@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch\;Qi�?!���L��?)\;Qi�?1���L��?:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat<��Ӹ7�?!�xH��$@)@OI��?11ɻ�\��?:Preprocessing2�
QIterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat::Prefetch::Mapo��ܚ��?!�8�h�UQ@)���T�~?1iS����?:Preprocessing2F
Iterator::Model@�R�?!��R[�@)�@�C�r?1��#�4��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 9.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t11.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9������"@I�/� �V@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�~�n��?�~�n��?!�~�n��?      ��!       "      ��!       *      ��!       2	�s�ڝ�?�s�ڝ�?!�s�ڝ�?:      ��!       B      ��!       J	�E&��H�?�E&��H�?!�E&��H�?R      ��!       Z	�E&��H�?�E&��H�?!�E&��H�?b      ��!       JCPU_ONLYY������"@b q�/� �V@