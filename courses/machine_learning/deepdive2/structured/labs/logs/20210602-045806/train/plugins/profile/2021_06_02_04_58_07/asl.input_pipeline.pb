	�9�w��?�9�w��?!�9�w��?	�.�L�!@�.�L�!@!�.�L�!@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�9�w��?���ި�?A�X�vM�?Y�#��:�?*	O��n���@2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat::Prefetch::Map::BatchV2::ShuffleAndRepeat::LegacyParallelInterleaveV2 <��kP�?!�#[��R@)<��kP�?1�#[��R@:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat::Prefetch::Map::BatchV2::ShuffleAndRepeat::LegacyParallelInterleaveV2[0]::CSV �8�t��?!����+$@)�8�t��?1����+$@:Preprocessing2�
ZIterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat::Prefetch::Map::BatchV2[y����?!ᯭ��U@)~t��gy�?1�ՙR��@:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat::Prefetch::Map::BatchV2::ShuffleAndRepeat ��NH�?!��*0�S@)b���X��?1J�왑@:Preprocessing2�
LIterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat::Prefetch:���u�?!_E���@):���u�?1_E���@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelisma5��6ƞ?!�ҿ0sZ�?)j��{��?1������?:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch�_YiR
�?!Ɓx\���?)�_YiR
�?1Ɓx\���?:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeatI��Z�֧?!�fu�I@)>����?1�a1���?:Preprocessing2�
QIterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat::Prefetch::Map������?!����*U@)��J?��v?1~LO[�?:Preprocessing2F
Iterator::Model'��d�V�?!F�h�@)g���p<o?1)�B����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 8.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t17.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�.�L�!@I�%zj��V@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���ި�?���ި�?!���ި�?      ��!       "      ��!       *      ��!       2	�X�vM�?�X�vM�?!�X�vM�?:      ��!       B      ��!       J	�#��:�?�#��:�?!�#��:�?R      ��!       Z	�#��:�?�#��:�?!�#��:�?b      ��!       JCPU_ONLYY�.�L�!@b q�%zj��V@