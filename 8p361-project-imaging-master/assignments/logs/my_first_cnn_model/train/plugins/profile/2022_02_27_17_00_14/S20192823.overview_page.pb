?	?ׁsFT]@?ׁsFT]@!?ׁsFT]@	K? ?r ??K? ?r ??!K? ?r ??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?ׁsFT]@Tt$?????A??e?cI]@Y?|гY???*?????I?@)      ?=2g
0Iterator::Model::Prefetch::FlatMap[0]::GeneratorR????\@!)?s?a?X@)R????\@1)?s?a?X@:Preprocessing2F
Iterator::Model9??v????!?l?d$???)?q??????1??\????:Preprocessing2P
Iterator::Model::Prefetch??_?L??!W趶a??)??_?L??1W趶a??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap<?R?!?\@!ٵ?]??X@)F%u?k?1????Tg?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9K? ?r ??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Tt$?????Tt$?????!Tt$?????      ??!       "      ??!       *      ??!       2	??e?cI]@??e?cI]@!??e?cI]@:      ??!       B      ??!       J	?|гY????|гY???!?|гY???R      ??!       Z	?|гY????|гY???!?|гY???JCPU_ONLYYK? ?r ??b Y      Y@q/?mX?x??"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 