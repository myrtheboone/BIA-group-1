	?w??#?s@?w??#?s@!?w??#?s@	3?]7P@3?]7P@!3?]7P@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?w??#?s@?[ A?c??AX9??v\@Y}??b?i@*	ffffVE	A2P
Iterator::Model::Prefetch.?!???i@!?ۛ?b?X@).?!???i@1?ۛ?b?X@:Preprocessing2F
Iterator::ModelF%u??i@!      Y@)8??d?`??1?Ƌ?̯??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 64.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no93?]7P@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?[ A?c???[ A?c??!?[ A?c??      ??!       "      ??!       *      ??!       2	X9??v\@X9??v\@!X9??v\@:      ??!       B      ??!       J	}??b?i@}??b?i@!}??b?i@R      ??!       Z	}??b?i@}??b?i@!}??b?i@JCPU_ONLYY3?]7P@b 