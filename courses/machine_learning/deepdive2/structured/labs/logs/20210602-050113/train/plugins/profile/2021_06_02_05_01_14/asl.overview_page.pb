�	V�Z� @V�Z� @!V�Z� @	lu�^@lu�^@!lu�^@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$V�Z� @�πz3j�?A��M(D �?Ym��?*	Q��nru�@2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat::Prefetch::Map::BatchV2::ShuffleAndRepeat::LegacyParallelInterleaveV2 m��~�b@!�HE�nS@)m��~�b@1�HE�nS@:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat::Prefetch::Map::BatchV2::ShuffleAndRepeat::LegacyParallelInterleaveV2[2]::CSV�h9�C��?!� (Jc&@)�h9�C��?1� (Jc&@:Preprocessing2�
�Iterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat::Prefetch::Map::BatchV2::ShuffleAndRepeat::LegacyParallelInterleaveV2[0]::CSV ����?�?!ƸUgC@)����?�?1ƸUgC@:Preprocessing2�
ZIterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat::Prefetch::Map::BatchV2�~��@!'[T@)���.�?1�+Qk~��?:Preprocessing2�
lIterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat::Prefetch::Map::BatchV2::ShuffleAndRepeat :ZՒ@!T���S@)^��j��?1U�#�q�?:Preprocessing2�
LIterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat::Prefetch�{)<hv�?!N.�y��?)�{)<hv�?1N.�y��?:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchj4��?!��A`O`�?)j4��?1��A`O`�?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism�&�Ҩ�?!�lwD�N�?)��<��?1��YQbz�?:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat
MK�ݳ?!m3!p�F�?){�Fw;�?1�(H�_o�?:Preprocessing2�
QIterator::Model::MaxIntraOpParallelism::Prefetch::ShuffleAndRepeat::Prefetch::Map�� �@!{`�mbT@)���|	u?1
��=�һ?:Preprocessing2F
Iterator::Model��`<�?!`�'F��?)V��6o�t?1c��B�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 18.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9ku�^@I��j��W@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�πz3j�?�πz3j�?!�πz3j�?      ��!       "      ��!       *      ��!       2	��M(D �?��M(D �?!��M(D �?:      ��!       B      ��!       J	m��?m��?!m��?R      ��!       Z	m��?m��?!m��?b      ��!       JCPU_ONLYYku�^@b q��j��W@Y      Y@q���0#@"�
both�Your program is POTENTIALLY input-bound because 18.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 