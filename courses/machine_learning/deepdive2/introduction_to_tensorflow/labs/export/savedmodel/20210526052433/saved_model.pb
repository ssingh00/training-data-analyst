�	
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.1.32v2.1.3-0-g77f47d68��
�
sequential/hidden1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: **
shared_namesequential/hidden1/kernel
�
-sequential/hidden1/kernel/Read/ReadVariableOpReadVariableOpsequential/hidden1/kernel*
_output_shapes

: *
dtype0
�
sequential/hidden1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namesequential/hidden1/bias

+sequential/hidden1/bias/Read/ReadVariableOpReadVariableOpsequential/hidden1/bias*
_output_shapes
: *
dtype0
�
sequential/hidden2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: **
shared_namesequential/hidden2/kernel
�
-sequential/hidden2/kernel/Read/ReadVariableOpReadVariableOpsequential/hidden2/kernel*
_output_shapes

: *
dtype0
�
sequential/hidden2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namesequential/hidden2/bias

+sequential/hidden2/bias/Read/ReadVariableOpReadVariableOpsequential/hidden2/bias*
_output_shapes
:*
dtype0
�
sequential/output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_namesequential/output/kernel
�
,sequential/output/kernel/Read/ReadVariableOpReadVariableOpsequential/output/kernel*
_output_shapes

:*
dtype0
�
sequential/output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namesequential/output/bias
}
*sequential/output/bias/Read/ReadVariableOpReadVariableOpsequential/output/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
 Adam/sequential/hidden1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *1
shared_name" Adam/sequential/hidden1/kernel/m
�
4Adam/sequential/hidden1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/sequential/hidden1/kernel/m*
_output_shapes

: *
dtype0
�
Adam/sequential/hidden1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/sequential/hidden1/bias/m
�
2Adam/sequential/hidden1/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/hidden1/bias/m*
_output_shapes
: *
dtype0
�
 Adam/sequential/hidden2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *1
shared_name" Adam/sequential/hidden2/kernel/m
�
4Adam/sequential/hidden2/kernel/m/Read/ReadVariableOpReadVariableOp Adam/sequential/hidden2/kernel/m*
_output_shapes

: *
dtype0
�
Adam/sequential/hidden2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/sequential/hidden2/bias/m
�
2Adam/sequential/hidden2/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/hidden2/bias/m*
_output_shapes
:*
dtype0
�
Adam/sequential/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Adam/sequential/output/kernel/m
�
3Adam/sequential/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/sequential/output/kernel/m*
_output_shapes

:*
dtype0
�
Adam/sequential/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/sequential/output/bias/m
�
1Adam/sequential/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/output/bias/m*
_output_shapes
:*
dtype0
�
 Adam/sequential/hidden1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *1
shared_name" Adam/sequential/hidden1/kernel/v
�
4Adam/sequential/hidden1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/sequential/hidden1/kernel/v*
_output_shapes

: *
dtype0
�
Adam/sequential/hidden1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name Adam/sequential/hidden1/bias/v
�
2Adam/sequential/hidden1/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/hidden1/bias/v*
_output_shapes
: *
dtype0
�
 Adam/sequential/hidden2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *1
shared_name" Adam/sequential/hidden2/kernel/v
�
4Adam/sequential/hidden2/kernel/v/Read/ReadVariableOpReadVariableOp Adam/sequential/hidden2/kernel/v*
_output_shapes

: *
dtype0
�
Adam/sequential/hidden2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/sequential/hidden2/bias/v
�
2Adam/sequential/hidden2/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/hidden2/bias/v*
_output_shapes
:*
dtype0
�
Adam/sequential/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*0
shared_name!Adam/sequential/output/kernel/v
�
3Adam/sequential/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/sequential/output/kernel/v*
_output_shapes

:*
dtype0
�
Adam/sequential/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/sequential/output/bias/v
�
1Adam/sequential/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�%
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�%
value�%B�% B�%
�
layer-0
layer-1
layer-2
layer-3
	optimizer
_training_endpoints
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
x
_feature_columns

_resources
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
�
$iter

%beta_1

&beta_2
	'decay
(learning_ratemImJmKmLmMmNvOvPvQvRvSvT
 
*
0
1
2
3
4
5
 
*
0
1
2
3
4
5
�
	variables

)layers
regularization_losses
*layer_regularization_losses
	trainable_variables
+non_trainable_variables
,metrics
 
 
 
 
 
 
�

-layers
	variables
.layer_regularization_losses
regularization_losses
trainable_variables
/non_trainable_variables
0metrics
XV
VARIABLE_VALUEsequential/hidden1/kernel)layer-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEsequential/hidden1/bias'layer-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�

1layers
	variables
2layer_regularization_losses
regularization_losses
trainable_variables
3non_trainable_variables
4metrics
XV
VARIABLE_VALUEsequential/hidden2/kernel)layer-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEsequential/hidden2/bias'layer-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�

5layers
	variables
6layer_regularization_losses
regularization_losses
trainable_variables
7non_trainable_variables
8metrics
WU
VARIABLE_VALUEsequential/output/kernel)layer-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEsequential/output/bias'layer-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�

9layers
 	variables
:layer_regularization_losses
!regularization_losses
"trainable_variables
;non_trainable_variables
<metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3
 
 

=0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
x
	>total
	?count
@
_fn_kwargs
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

>0
?1
 
 
�

Elayers
A	variables
Flayer_regularization_losses
Bregularization_losses
Ctrainable_variables
Gnon_trainable_variables
Hmetrics
 
 

>0
?1
 
{y
VARIABLE_VALUE Adam/sequential/hidden1/kernel/mElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/sequential/hidden1/bias/mClayer-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE Adam/sequential/hidden2/kernel/mElayer-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/sequential/hidden2/bias/mClayer-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/sequential/output/kernel/mElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/sequential/output/bias/mClayer-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE Adam/sequential/hidden1/kernel/vElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/sequential/hidden1/bias/vClayer-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE Adam/sequential/hidden2/kernel/vElayer-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/sequential/hidden2/bias/vClayer-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/sequential/output/kernel/vElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/sequential/output/bias/vClayer-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
 serving_default_dropoff_latitudePlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
!serving_default_dropoff_longitudePlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
serving_default_passenger_countPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
serving_default_pickup_latitudePlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
 serving_default_pickup_longitudePlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCall serving_default_dropoff_latitude!serving_default_dropoff_longitudeserving_default_passenger_countserving_default_pickup_latitude serving_default_pickup_longitudesequential/hidden1/kernelsequential/hidden1/biassequential/hidden2/kernelsequential/hidden2/biassequential/output/kernelsequential/output/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference_signature_wrapper_5543
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-sequential/hidden1/kernel/Read/ReadVariableOp+sequential/hidden1/bias/Read/ReadVariableOp-sequential/hidden2/kernel/Read/ReadVariableOp+sequential/hidden2/bias/Read/ReadVariableOp,sequential/output/kernel/Read/ReadVariableOp*sequential/output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp4Adam/sequential/hidden1/kernel/m/Read/ReadVariableOp2Adam/sequential/hidden1/bias/m/Read/ReadVariableOp4Adam/sequential/hidden2/kernel/m/Read/ReadVariableOp2Adam/sequential/hidden2/bias/m/Read/ReadVariableOp3Adam/sequential/output/kernel/m/Read/ReadVariableOp1Adam/sequential/output/bias/m/Read/ReadVariableOp4Adam/sequential/hidden1/kernel/v/Read/ReadVariableOp2Adam/sequential/hidden1/bias/v/Read/ReadVariableOp4Adam/sequential/hidden2/kernel/v/Read/ReadVariableOp2Adam/sequential/hidden2/bias/v/Read/ReadVariableOp3Adam/sequential/output/kernel/v/Read/ReadVariableOp1Adam/sequential/output/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*&
f!R
__inference__traced_save_5928
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesequential/hidden1/kernelsequential/hidden1/biassequential/hidden2/kernelsequential/hidden2/biassequential/output/kernelsequential/output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount Adam/sequential/hidden1/kernel/mAdam/sequential/hidden1/bias/m Adam/sequential/hidden2/kernel/mAdam/sequential/hidden2/bias/mAdam/sequential/output/kernel/mAdam/sequential/output/bias/m Adam/sequential/hidden1/kernel/vAdam/sequential/hidden1/bias/v Adam/sequential/hidden2/kernel/vAdam/sequential/hidden2/bias/vAdam/sequential/output/kernel/vAdam/sequential/output/bias/v*%
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__traced_restore_6015��
�
�
)__inference_sequential_layer_call_fn_5486
dropoff_latitude
dropoff_longitude
passenger_count
pickup_latitude
pickup_longitude"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldropoff_latitudedropoff_longitudepassenger_countpickup_latitudepickup_longitudestatefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_54772
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesy
w:���������:���������:���������:���������:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:0 ,
*
_user_specified_namedropoff_latitude:1-
+
_user_specified_namedropoff_longitude:/+
)
_user_specified_namepassenger_count:/+
)
_user_specified_namepickup_latitude:0,
*
_user_specified_namepickup_longitude
�
�
D__inference_sequential_layer_call_and_return_conditional_losses_5434
dropoff_latitude
dropoff_longitude
passenger_count
pickup_latitude
pickup_longitude*
&hidden1_statefulpartitionedcall_args_1*
&hidden1_statefulpartitionedcall_args_2*
&hidden2_statefulpartitionedcall_args_1*
&hidden2_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity��hidden1/StatefulPartitionedCall�hidden2/StatefulPartitionedCall�output/StatefulPartitionedCall�
 dense_features_1/PartitionedCallPartitionedCalldropoff_latitudedropoff_longitudepassenger_countpickup_latitudepickup_longitude*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_dense_features_1_layer_call_and_return_conditional_losses_53532"
 dense_features_1/PartitionedCall�
hidden1/StatefulPartitionedCallStatefulPartitionedCall)dense_features_1/PartitionedCall:output:0&hidden1_statefulpartitionedcall_args_1&hidden1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:��������� **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_hidden1_layer_call_and_return_conditional_losses_53762!
hidden1/StatefulPartitionedCall�
hidden2/StatefulPartitionedCallStatefulPartitionedCall(hidden1/StatefulPartitionedCall:output:0&hidden2_statefulpartitionedcall_args_1&hidden2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_hidden2_layer_call_and_return_conditional_losses_53992!
hidden2/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall(hidden2/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_54212 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^hidden1/StatefulPartitionedCall ^hidden2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesy
w:���������:���������:���������:���������:���������::::::2B
hidden1/StatefulPartitionedCallhidden1/StatefulPartitionedCall2B
hidden2/StatefulPartitionedCallhidden2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:0 ,
*
_user_specified_namedropoff_latitude:1-
+
_user_specified_namedropoff_longitude:/+
)
_user_specified_namepassenger_count:/+
)
_user_specified_namepickup_latitude:0,
*
_user_specified_namepickup_longitude
�9
�
__inference__traced_save_5928
file_prefix8
4savev2_sequential_hidden1_kernel_read_readvariableop6
2savev2_sequential_hidden1_bias_read_readvariableop8
4savev2_sequential_hidden2_kernel_read_readvariableop6
2savev2_sequential_hidden2_bias_read_readvariableop7
3savev2_sequential_output_kernel_read_readvariableop5
1savev2_sequential_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop?
;savev2_adam_sequential_hidden1_kernel_m_read_readvariableop=
9savev2_adam_sequential_hidden1_bias_m_read_readvariableop?
;savev2_adam_sequential_hidden2_kernel_m_read_readvariableop=
9savev2_adam_sequential_hidden2_bias_m_read_readvariableop>
:savev2_adam_sequential_output_kernel_m_read_readvariableop<
8savev2_adam_sequential_output_bias_m_read_readvariableop?
;savev2_adam_sequential_hidden1_kernel_v_read_readvariableop=
9savev2_adam_sequential_hidden1_bias_v_read_readvariableop?
;savev2_adam_sequential_hidden2_kernel_v_read_readvariableop=
9savev2_adam_sequential_hidden2_bias_v_read_readvariableop>
:savev2_adam_sequential_output_kernel_v_read_readvariableop<
8savev2_adam_sequential_output_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_bedf14960e114619856a85796d862653/part2
StringJoin/inputs_1�

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B)layer-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)layer-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)layer-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBElayer-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_sequential_hidden1_kernel_read_readvariableop2savev2_sequential_hidden1_bias_read_readvariableop4savev2_sequential_hidden2_kernel_read_readvariableop2savev2_sequential_hidden2_bias_read_readvariableop3savev2_sequential_output_kernel_read_readvariableop1savev2_sequential_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop;savev2_adam_sequential_hidden1_kernel_m_read_readvariableop9savev2_adam_sequential_hidden1_bias_m_read_readvariableop;savev2_adam_sequential_hidden2_kernel_m_read_readvariableop9savev2_adam_sequential_hidden2_bias_m_read_readvariableop:savev2_adam_sequential_output_kernel_m_read_readvariableop8savev2_adam_sequential_output_bias_m_read_readvariableop;savev2_adam_sequential_hidden1_kernel_v_read_readvariableop9savev2_adam_sequential_hidden1_bias_v_read_readvariableop;savev2_adam_sequential_hidden2_kernel_v_read_readvariableop9savev2_adam_sequential_hidden2_bias_v_read_readvariableop:savev2_adam_sequential_output_kernel_v_read_readvariableop8savev2_adam_sequential_output_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *'
dtypes
2	2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : :::: : : : : : : : : : :::: : : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�
�
D__inference_sequential_layer_call_and_return_conditional_losses_5452
dropoff_latitude
dropoff_longitude
passenger_count
pickup_latitude
pickup_longitude*
&hidden1_statefulpartitionedcall_args_1*
&hidden1_statefulpartitionedcall_args_2*
&hidden2_statefulpartitionedcall_args_1*
&hidden2_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity��hidden1/StatefulPartitionedCall�hidden2/StatefulPartitionedCall�output/StatefulPartitionedCall�
 dense_features_1/PartitionedCallPartitionedCalldropoff_latitudedropoff_longitudepassenger_countpickup_latitudepickup_longitude*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_dense_features_1_layer_call_and_return_conditional_losses_53532"
 dense_features_1/PartitionedCall�
hidden1/StatefulPartitionedCallStatefulPartitionedCall)dense_features_1/PartitionedCall:output:0&hidden1_statefulpartitionedcall_args_1&hidden1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:��������� **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_hidden1_layer_call_and_return_conditional_losses_53762!
hidden1/StatefulPartitionedCall�
hidden2/StatefulPartitionedCallStatefulPartitionedCall(hidden1/StatefulPartitionedCall:output:0&hidden2_statefulpartitionedcall_args_1&hidden2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_hidden2_layer_call_and_return_conditional_losses_53992!
hidden2/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall(hidden2/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_54212 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^hidden1/StatefulPartitionedCall ^hidden2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesy
w:���������:���������:���������:���������:���������::::::2B
hidden1/StatefulPartitionedCallhidden1/StatefulPartitionedCall2B
hidden2/StatefulPartitionedCallhidden2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:0 ,
*
_user_specified_namedropoff_latitude:1-
+
_user_specified_namedropoff_longitude:/+
)
_user_specified_namepassenger_count:/+
)
_user_specified_namepickup_latitude:0,
*
_user_specified_namepickup_longitude
�?
�
J__inference_dense_features_1_layer_call_and_return_conditional_losses_5353
features

features_1

features_2

features_3

features_4
identityh
dropoff_latitude/ShapeShapefeatures*
T0*
_output_shapes
:2
dropoff_latitude/Shape�
$dropoff_latitude/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$dropoff_latitude/strided_slice/stack�
&dropoff_latitude/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&dropoff_latitude/strided_slice/stack_1�
&dropoff_latitude/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&dropoff_latitude/strided_slice/stack_2�
dropoff_latitude/strided_sliceStridedSlicedropoff_latitude/Shape:output:0-dropoff_latitude/strided_slice/stack:output:0/dropoff_latitude/strided_slice/stack_1:output:0/dropoff_latitude/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
dropoff_latitude/strided_slice�
 dropoff_latitude/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 dropoff_latitude/Reshape/shape/1�
dropoff_latitude/Reshape/shapePack'dropoff_latitude/strided_slice:output:0)dropoff_latitude/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
dropoff_latitude/Reshape/shape�
dropoff_latitude/ReshapeReshapefeatures'dropoff_latitude/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2
dropoff_latitude/Reshapel
dropoff_longitude/ShapeShape
features_1*
T0*
_output_shapes
:2
dropoff_longitude/Shape�
%dropoff_longitude/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%dropoff_longitude/strided_slice/stack�
'dropoff_longitude/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'dropoff_longitude/strided_slice/stack_1�
'dropoff_longitude/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'dropoff_longitude/strided_slice/stack_2�
dropoff_longitude/strided_sliceStridedSlice dropoff_longitude/Shape:output:0.dropoff_longitude/strided_slice/stack:output:00dropoff_longitude/strided_slice/stack_1:output:00dropoff_longitude/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
dropoff_longitude/strided_slice�
!dropoff_longitude/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!dropoff_longitude/Reshape/shape/1�
dropoff_longitude/Reshape/shapePack(dropoff_longitude/strided_slice:output:0*dropoff_longitude/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
dropoff_longitude/Reshape/shape�
dropoff_longitude/ReshapeReshape
features_1(dropoff_longitude/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2
dropoff_longitude/Reshapeh
passenger_count/ShapeShape
features_2*
T0*
_output_shapes
:2
passenger_count/Shape�
#passenger_count/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#passenger_count/strided_slice/stack�
%passenger_count/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%passenger_count/strided_slice/stack_1�
%passenger_count/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%passenger_count/strided_slice/stack_2�
passenger_count/strided_sliceStridedSlicepassenger_count/Shape:output:0,passenger_count/strided_slice/stack:output:0.passenger_count/strided_slice/stack_1:output:0.passenger_count/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
passenger_count/strided_slice�
passenger_count/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
passenger_count/Reshape/shape/1�
passenger_count/Reshape/shapePack&passenger_count/strided_slice:output:0(passenger_count/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
passenger_count/Reshape/shape�
passenger_count/ReshapeReshape
features_2&passenger_count/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2
passenger_count/Reshapeh
pickup_latitude/ShapeShape
features_3*
T0*
_output_shapes
:2
pickup_latitude/Shape�
#pickup_latitude/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#pickup_latitude/strided_slice/stack�
%pickup_latitude/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%pickup_latitude/strided_slice/stack_1�
%pickup_latitude/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%pickup_latitude/strided_slice/stack_2�
pickup_latitude/strided_sliceStridedSlicepickup_latitude/Shape:output:0,pickup_latitude/strided_slice/stack:output:0.pickup_latitude/strided_slice/stack_1:output:0.pickup_latitude/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
pickup_latitude/strided_slice�
pickup_latitude/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
pickup_latitude/Reshape/shape/1�
pickup_latitude/Reshape/shapePack&pickup_latitude/strided_slice:output:0(pickup_latitude/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
pickup_latitude/Reshape/shape�
pickup_latitude/ReshapeReshape
features_3&pickup_latitude/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2
pickup_latitude/Reshapej
pickup_longitude/ShapeShape
features_4*
T0*
_output_shapes
:2
pickup_longitude/Shape�
$pickup_longitude/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$pickup_longitude/strided_slice/stack�
&pickup_longitude/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&pickup_longitude/strided_slice/stack_1�
&pickup_longitude/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&pickup_longitude/strided_slice/stack_2�
pickup_longitude/strided_sliceStridedSlicepickup_longitude/Shape:output:0-pickup_longitude/strided_slice/stack:output:0/pickup_longitude/strided_slice/stack_1:output:0/pickup_longitude/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
pickup_longitude/strided_slice�
 pickup_longitude/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 pickup_longitude/Reshape/shape/1�
pickup_longitude/Reshape/shapePack'pickup_longitude/strided_slice:output:0)pickup_longitude/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
pickup_longitude/Reshape/shape�
pickup_longitude/ReshapeReshape
features_4'pickup_longitude/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2
pickup_longitude/Reshapee
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat/axis�
concatConcatV2!dropoff_latitude/Reshape:output:0"dropoff_longitude/Reshape:output:0 passenger_count/Reshape:output:0 pickup_latitude/Reshape:output:0!pickup_longitude/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:���������:���������:���������:���������:���������:( $
"
_user_specified_name
features:($
"
_user_specified_name
features:($
"
_user_specified_name
features:($
"
_user_specified_name
features:($
"
_user_specified_name
features
�
�
&__inference_hidden2_layer_call_fn_5808

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_hidden2_layer_call_and_return_conditional_losses_53992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�{
�
__inference__wrapped_model_5295
dropoff_latitude
dropoff_longitude
passenger_count
pickup_latitude
pickup_longitude5
1sequential_hidden1_matmul_readvariableop_resource6
2sequential_hidden1_biasadd_readvariableop_resource5
1sequential_hidden2_matmul_readvariableop_resource6
2sequential_hidden2_biasadd_readvariableop_resource4
0sequential_output_matmul_readvariableop_resource5
1sequential_output_biasadd_readvariableop_resource
identity��)sequential/hidden1/BiasAdd/ReadVariableOp�(sequential/hidden1/MatMul/ReadVariableOp�)sequential/hidden2/BiasAdd/ReadVariableOp�(sequential/hidden2/MatMul/ReadVariableOp�(sequential/output/BiasAdd/ReadVariableOp�'sequential/output/MatMul/ReadVariableOp�
2sequential/dense_features_1/dropoff_latitude/ShapeShapedropoff_latitude*
T0*
_output_shapes
:24
2sequential/dense_features_1/dropoff_latitude/Shape�
@sequential/dense_features_1/dropoff_latitude/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential/dense_features_1/dropoff_latitude/strided_slice/stack�
Bsequential/dense_features_1/dropoff_latitude/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential/dense_features_1/dropoff_latitude/strided_slice/stack_1�
Bsequential/dense_features_1/dropoff_latitude/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential/dense_features_1/dropoff_latitude/strided_slice/stack_2�
:sequential/dense_features_1/dropoff_latitude/strided_sliceStridedSlice;sequential/dense_features_1/dropoff_latitude/Shape:output:0Isequential/dense_features_1/dropoff_latitude/strided_slice/stack:output:0Ksequential/dense_features_1/dropoff_latitude/strided_slice/stack_1:output:0Ksequential/dense_features_1/dropoff_latitude/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential/dense_features_1/dropoff_latitude/strided_slice�
<sequential/dense_features_1/dropoff_latitude/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2>
<sequential/dense_features_1/dropoff_latitude/Reshape/shape/1�
:sequential/dense_features_1/dropoff_latitude/Reshape/shapePackCsequential/dense_features_1/dropoff_latitude/strided_slice:output:0Esequential/dense_features_1/dropoff_latitude/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2<
:sequential/dense_features_1/dropoff_latitude/Reshape/shape�
4sequential/dense_features_1/dropoff_latitude/ReshapeReshapedropoff_latitudeCsequential/dense_features_1/dropoff_latitude/Reshape/shape:output:0*
T0*'
_output_shapes
:���������26
4sequential/dense_features_1/dropoff_latitude/Reshape�
3sequential/dense_features_1/dropoff_longitude/ShapeShapedropoff_longitude*
T0*
_output_shapes
:25
3sequential/dense_features_1/dropoff_longitude/Shape�
Asequential/dense_features_1/dropoff_longitude/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential/dense_features_1/dropoff_longitude/strided_slice/stack�
Csequential/dense_features_1/dropoff_longitude/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential/dense_features_1/dropoff_longitude/strided_slice/stack_1�
Csequential/dense_features_1/dropoff_longitude/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential/dense_features_1/dropoff_longitude/strided_slice/stack_2�
;sequential/dense_features_1/dropoff_longitude/strided_sliceStridedSlice<sequential/dense_features_1/dropoff_longitude/Shape:output:0Jsequential/dense_features_1/dropoff_longitude/strided_slice/stack:output:0Lsequential/dense_features_1/dropoff_longitude/strided_slice/stack_1:output:0Lsequential/dense_features_1/dropoff_longitude/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;sequential/dense_features_1/dropoff_longitude/strided_slice�
=sequential/dense_features_1/dropoff_longitude/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2?
=sequential/dense_features_1/dropoff_longitude/Reshape/shape/1�
;sequential/dense_features_1/dropoff_longitude/Reshape/shapePackDsequential/dense_features_1/dropoff_longitude/strided_slice:output:0Fsequential/dense_features_1/dropoff_longitude/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2=
;sequential/dense_features_1/dropoff_longitude/Reshape/shape�
5sequential/dense_features_1/dropoff_longitude/ReshapeReshapedropoff_longitudeDsequential/dense_features_1/dropoff_longitude/Reshape/shape:output:0*
T0*'
_output_shapes
:���������27
5sequential/dense_features_1/dropoff_longitude/Reshape�
1sequential/dense_features_1/passenger_count/ShapeShapepassenger_count*
T0*
_output_shapes
:23
1sequential/dense_features_1/passenger_count/Shape�
?sequential/dense_features_1/passenger_count/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?sequential/dense_features_1/passenger_count/strided_slice/stack�
Asequential/dense_features_1/passenger_count/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Asequential/dense_features_1/passenger_count/strided_slice/stack_1�
Asequential/dense_features_1/passenger_count/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Asequential/dense_features_1/passenger_count/strided_slice/stack_2�
9sequential/dense_features_1/passenger_count/strided_sliceStridedSlice:sequential/dense_features_1/passenger_count/Shape:output:0Hsequential/dense_features_1/passenger_count/strided_slice/stack:output:0Jsequential/dense_features_1/passenger_count/strided_slice/stack_1:output:0Jsequential/dense_features_1/passenger_count/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9sequential/dense_features_1/passenger_count/strided_slice�
;sequential/dense_features_1/passenger_count/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2=
;sequential/dense_features_1/passenger_count/Reshape/shape/1�
9sequential/dense_features_1/passenger_count/Reshape/shapePackBsequential/dense_features_1/passenger_count/strided_slice:output:0Dsequential/dense_features_1/passenger_count/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2;
9sequential/dense_features_1/passenger_count/Reshape/shape�
3sequential/dense_features_1/passenger_count/ReshapeReshapepassenger_countBsequential/dense_features_1/passenger_count/Reshape/shape:output:0*
T0*'
_output_shapes
:���������25
3sequential/dense_features_1/passenger_count/Reshape�
1sequential/dense_features_1/pickup_latitude/ShapeShapepickup_latitude*
T0*
_output_shapes
:23
1sequential/dense_features_1/pickup_latitude/Shape�
?sequential/dense_features_1/pickup_latitude/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?sequential/dense_features_1/pickup_latitude/strided_slice/stack�
Asequential/dense_features_1/pickup_latitude/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Asequential/dense_features_1/pickup_latitude/strided_slice/stack_1�
Asequential/dense_features_1/pickup_latitude/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Asequential/dense_features_1/pickup_latitude/strided_slice/stack_2�
9sequential/dense_features_1/pickup_latitude/strided_sliceStridedSlice:sequential/dense_features_1/pickup_latitude/Shape:output:0Hsequential/dense_features_1/pickup_latitude/strided_slice/stack:output:0Jsequential/dense_features_1/pickup_latitude/strided_slice/stack_1:output:0Jsequential/dense_features_1/pickup_latitude/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9sequential/dense_features_1/pickup_latitude/strided_slice�
;sequential/dense_features_1/pickup_latitude/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2=
;sequential/dense_features_1/pickup_latitude/Reshape/shape/1�
9sequential/dense_features_1/pickup_latitude/Reshape/shapePackBsequential/dense_features_1/pickup_latitude/strided_slice:output:0Dsequential/dense_features_1/pickup_latitude/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2;
9sequential/dense_features_1/pickup_latitude/Reshape/shape�
3sequential/dense_features_1/pickup_latitude/ReshapeReshapepickup_latitudeBsequential/dense_features_1/pickup_latitude/Reshape/shape:output:0*
T0*'
_output_shapes
:���������25
3sequential/dense_features_1/pickup_latitude/Reshape�
2sequential/dense_features_1/pickup_longitude/ShapeShapepickup_longitude*
T0*
_output_shapes
:24
2sequential/dense_features_1/pickup_longitude/Shape�
@sequential/dense_features_1/pickup_longitude/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential/dense_features_1/pickup_longitude/strided_slice/stack�
Bsequential/dense_features_1/pickup_longitude/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential/dense_features_1/pickup_longitude/strided_slice/stack_1�
Bsequential/dense_features_1/pickup_longitude/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential/dense_features_1/pickup_longitude/strided_slice/stack_2�
:sequential/dense_features_1/pickup_longitude/strided_sliceStridedSlice;sequential/dense_features_1/pickup_longitude/Shape:output:0Isequential/dense_features_1/pickup_longitude/strided_slice/stack:output:0Ksequential/dense_features_1/pickup_longitude/strided_slice/stack_1:output:0Ksequential/dense_features_1/pickup_longitude/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential/dense_features_1/pickup_longitude/strided_slice�
<sequential/dense_features_1/pickup_longitude/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2>
<sequential/dense_features_1/pickup_longitude/Reshape/shape/1�
:sequential/dense_features_1/pickup_longitude/Reshape/shapePackCsequential/dense_features_1/pickup_longitude/strided_slice:output:0Esequential/dense_features_1/pickup_longitude/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2<
:sequential/dense_features_1/pickup_longitude/Reshape/shape�
4sequential/dense_features_1/pickup_longitude/ReshapeReshapepickup_longitudeCsequential/dense_features_1/pickup_longitude/Reshape/shape:output:0*
T0*'
_output_shapes
:���������26
4sequential/dense_features_1/pickup_longitude/Reshape�
'sequential/dense_features_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2)
'sequential/dense_features_1/concat/axis�
"sequential/dense_features_1/concatConcatV2=sequential/dense_features_1/dropoff_latitude/Reshape:output:0>sequential/dense_features_1/dropoff_longitude/Reshape:output:0<sequential/dense_features_1/passenger_count/Reshape:output:0<sequential/dense_features_1/pickup_latitude/Reshape:output:0=sequential/dense_features_1/pickup_longitude/Reshape:output:00sequential/dense_features_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2$
"sequential/dense_features_1/concat�
(sequential/hidden1/MatMul/ReadVariableOpReadVariableOp1sequential_hidden1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02*
(sequential/hidden1/MatMul/ReadVariableOp�
sequential/hidden1/MatMulMatMul+sequential/dense_features_1/concat:output:00sequential/hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
sequential/hidden1/MatMul�
)sequential/hidden1/BiasAdd/ReadVariableOpReadVariableOp2sequential_hidden1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)sequential/hidden1/BiasAdd/ReadVariableOp�
sequential/hidden1/BiasAddBiasAdd#sequential/hidden1/MatMul:product:01sequential/hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
sequential/hidden1/BiasAdd�
sequential/hidden1/ReluRelu#sequential/hidden1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
sequential/hidden1/Relu�
(sequential/hidden2/MatMul/ReadVariableOpReadVariableOp1sequential_hidden2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02*
(sequential/hidden2/MatMul/ReadVariableOp�
sequential/hidden2/MatMulMatMul%sequential/hidden1/Relu:activations:00sequential/hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential/hidden2/MatMul�
)sequential/hidden2/BiasAdd/ReadVariableOpReadVariableOp2sequential_hidden2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/hidden2/BiasAdd/ReadVariableOp�
sequential/hidden2/BiasAddBiasAdd#sequential/hidden2/MatMul:product:01sequential/hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential/hidden2/BiasAdd�
sequential/hidden2/ReluRelu#sequential/hidden2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
sequential/hidden2/Relu�
'sequential/output/MatMul/ReadVariableOpReadVariableOp0sequential_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'sequential/output/MatMul/ReadVariableOp�
sequential/output/MatMulMatMul%sequential/hidden2/Relu:activations:0/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential/output/MatMul�
(sequential/output/BiasAdd/ReadVariableOpReadVariableOp1sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/output/BiasAdd/ReadVariableOp�
sequential/output/BiasAddBiasAdd"sequential/output/MatMul:product:00sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential/output/BiasAdd�
IdentityIdentity"sequential/output/BiasAdd:output:0*^sequential/hidden1/BiasAdd/ReadVariableOp)^sequential/hidden1/MatMul/ReadVariableOp*^sequential/hidden2/BiasAdd/ReadVariableOp)^sequential/hidden2/MatMul/ReadVariableOp)^sequential/output/BiasAdd/ReadVariableOp(^sequential/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesy
w:���������:���������:���������:���������:���������::::::2V
)sequential/hidden1/BiasAdd/ReadVariableOp)sequential/hidden1/BiasAdd/ReadVariableOp2T
(sequential/hidden1/MatMul/ReadVariableOp(sequential/hidden1/MatMul/ReadVariableOp2V
)sequential/hidden2/BiasAdd/ReadVariableOp)sequential/hidden2/BiasAdd/ReadVariableOp2T
(sequential/hidden2/MatMul/ReadVariableOp(sequential/hidden2/MatMul/ReadVariableOp2T
(sequential/output/BiasAdd/ReadVariableOp(sequential/output/BiasAdd/ReadVariableOp2R
'sequential/output/MatMul/ReadVariableOp'sequential/output/MatMul/ReadVariableOp:0 ,
*
_user_specified_namedropoff_latitude:1-
+
_user_specified_namedropoff_longitude:/+
)
_user_specified_namepassenger_count:/+
)
_user_specified_namepickup_latitude:0,
*
_user_specified_namepickup_longitude
�	
�
/__inference_dense_features_1_layer_call_fn_5772
features_dropoff_latitude
features_dropoff_longitude
features_passenger_count
features_pickup_latitude
features_pickup_longitude
identity�
PartitionedCallPartitionedCallfeatures_dropoff_latitudefeatures_dropoff_longitudefeatures_passenger_countfeatures_pickup_latitudefeatures_pickup_longitude*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_dense_features_1_layer_call_and_return_conditional_losses_53532
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:���������:���������:���������:���������:���������:9 5
3
_user_specified_namefeatures/dropoff_latitude::6
4
_user_specified_namefeatures/dropoff_longitude:84
2
_user_specified_namefeatures/passenger_count:84
2
_user_specified_namefeatures/pickup_latitude:95
3
_user_specified_namefeatures/pickup_longitude
�
�
)__inference_sequential_layer_call_fn_5698
inputs_dropoff_latitude
inputs_dropoff_longitude
inputs_passenger_count
inputs_pickup_latitude
inputs_pickup_longitude"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_dropoff_latitudeinputs_dropoff_longitudeinputs_passenger_countinputs_pickup_latitudeinputs_pickup_longitudestatefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_54772
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesy
w:���������:���������:���������:���������:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:7 3
1
_user_specified_nameinputs/dropoff_latitude:84
2
_user_specified_nameinputs/dropoff_longitude:62
0
_user_specified_nameinputs/passenger_count:62
0
_user_specified_nameinputs/pickup_latitude:73
1
_user_specified_nameinputs/pickup_longitude
�	
�
A__inference_hidden1_layer_call_and_return_conditional_losses_5783

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�k
�
D__inference_sequential_layer_call_and_return_conditional_losses_5613
inputs_dropoff_latitude
inputs_dropoff_longitude
inputs_passenger_count
inputs_pickup_latitude
inputs_pickup_longitude*
&hidden1_matmul_readvariableop_resource+
'hidden1_biasadd_readvariableop_resource*
&hidden2_matmul_readvariableop_resource+
'hidden2_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity��hidden1/BiasAdd/ReadVariableOp�hidden1/MatMul/ReadVariableOp�hidden2/BiasAdd/ReadVariableOp�hidden2/MatMul/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOp�
'dense_features_1/dropoff_latitude/ShapeShapeinputs_dropoff_latitude*
T0*
_output_shapes
:2)
'dense_features_1/dropoff_latitude/Shape�
5dense_features_1/dropoff_latitude/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5dense_features_1/dropoff_latitude/strided_slice/stack�
7dense_features_1/dropoff_latitude/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/dropoff_latitude/strided_slice/stack_1�
7dense_features_1/dropoff_latitude/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/dropoff_latitude/strided_slice/stack_2�
/dense_features_1/dropoff_latitude/strided_sliceStridedSlice0dense_features_1/dropoff_latitude/Shape:output:0>dense_features_1/dropoff_latitude/strided_slice/stack:output:0@dense_features_1/dropoff_latitude/strided_slice/stack_1:output:0@dense_features_1/dropoff_latitude/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/dense_features_1/dropoff_latitude/strided_slice�
1dense_features_1/dropoff_latitude/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1dense_features_1/dropoff_latitude/Reshape/shape/1�
/dense_features_1/dropoff_latitude/Reshape/shapePack8dense_features_1/dropoff_latitude/strided_slice:output:0:dense_features_1/dropoff_latitude/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:21
/dense_features_1/dropoff_latitude/Reshape/shape�
)dense_features_1/dropoff_latitude/ReshapeReshapeinputs_dropoff_latitude8dense_features_1/dropoff_latitude/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2+
)dense_features_1/dropoff_latitude/Reshape�
(dense_features_1/dropoff_longitude/ShapeShapeinputs_dropoff_longitude*
T0*
_output_shapes
:2*
(dense_features_1/dropoff_longitude/Shape�
6dense_features_1/dropoff_longitude/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6dense_features_1/dropoff_longitude/strided_slice/stack�
8dense_features_1/dropoff_longitude/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8dense_features_1/dropoff_longitude/strided_slice/stack_1�
8dense_features_1/dropoff_longitude/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8dense_features_1/dropoff_longitude/strided_slice/stack_2�
0dense_features_1/dropoff_longitude/strided_sliceStridedSlice1dense_features_1/dropoff_longitude/Shape:output:0?dense_features_1/dropoff_longitude/strided_slice/stack:output:0Adense_features_1/dropoff_longitude/strided_slice/stack_1:output:0Adense_features_1/dropoff_longitude/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0dense_features_1/dropoff_longitude/strided_slice�
2dense_features_1/dropoff_longitude/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :24
2dense_features_1/dropoff_longitude/Reshape/shape/1�
0dense_features_1/dropoff_longitude/Reshape/shapePack9dense_features_1/dropoff_longitude/strided_slice:output:0;dense_features_1/dropoff_longitude/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:22
0dense_features_1/dropoff_longitude/Reshape/shape�
*dense_features_1/dropoff_longitude/ReshapeReshapeinputs_dropoff_longitude9dense_features_1/dropoff_longitude/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2,
*dense_features_1/dropoff_longitude/Reshape�
&dense_features_1/passenger_count/ShapeShapeinputs_passenger_count*
T0*
_output_shapes
:2(
&dense_features_1/passenger_count/Shape�
4dense_features_1/passenger_count/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4dense_features_1/passenger_count/strided_slice/stack�
6dense_features_1/passenger_count/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6dense_features_1/passenger_count/strided_slice/stack_1�
6dense_features_1/passenger_count/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6dense_features_1/passenger_count/strided_slice/stack_2�
.dense_features_1/passenger_count/strided_sliceStridedSlice/dense_features_1/passenger_count/Shape:output:0=dense_features_1/passenger_count/strided_slice/stack:output:0?dense_features_1/passenger_count/strided_slice/stack_1:output:0?dense_features_1/passenger_count/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.dense_features_1/passenger_count/strided_slice�
0dense_features_1/passenger_count/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0dense_features_1/passenger_count/Reshape/shape/1�
.dense_features_1/passenger_count/Reshape/shapePack7dense_features_1/passenger_count/strided_slice:output:09dense_features_1/passenger_count/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:20
.dense_features_1/passenger_count/Reshape/shape�
(dense_features_1/passenger_count/ReshapeReshapeinputs_passenger_count7dense_features_1/passenger_count/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2*
(dense_features_1/passenger_count/Reshape�
&dense_features_1/pickup_latitude/ShapeShapeinputs_pickup_latitude*
T0*
_output_shapes
:2(
&dense_features_1/pickup_latitude/Shape�
4dense_features_1/pickup_latitude/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4dense_features_1/pickup_latitude/strided_slice/stack�
6dense_features_1/pickup_latitude/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6dense_features_1/pickup_latitude/strided_slice/stack_1�
6dense_features_1/pickup_latitude/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6dense_features_1/pickup_latitude/strided_slice/stack_2�
.dense_features_1/pickup_latitude/strided_sliceStridedSlice/dense_features_1/pickup_latitude/Shape:output:0=dense_features_1/pickup_latitude/strided_slice/stack:output:0?dense_features_1/pickup_latitude/strided_slice/stack_1:output:0?dense_features_1/pickup_latitude/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.dense_features_1/pickup_latitude/strided_slice�
0dense_features_1/pickup_latitude/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0dense_features_1/pickup_latitude/Reshape/shape/1�
.dense_features_1/pickup_latitude/Reshape/shapePack7dense_features_1/pickup_latitude/strided_slice:output:09dense_features_1/pickup_latitude/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:20
.dense_features_1/pickup_latitude/Reshape/shape�
(dense_features_1/pickup_latitude/ReshapeReshapeinputs_pickup_latitude7dense_features_1/pickup_latitude/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2*
(dense_features_1/pickup_latitude/Reshape�
'dense_features_1/pickup_longitude/ShapeShapeinputs_pickup_longitude*
T0*
_output_shapes
:2)
'dense_features_1/pickup_longitude/Shape�
5dense_features_1/pickup_longitude/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5dense_features_1/pickup_longitude/strided_slice/stack�
7dense_features_1/pickup_longitude/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/pickup_longitude/strided_slice/stack_1�
7dense_features_1/pickup_longitude/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/pickup_longitude/strided_slice/stack_2�
/dense_features_1/pickup_longitude/strided_sliceStridedSlice0dense_features_1/pickup_longitude/Shape:output:0>dense_features_1/pickup_longitude/strided_slice/stack:output:0@dense_features_1/pickup_longitude/strided_slice/stack_1:output:0@dense_features_1/pickup_longitude/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/dense_features_1/pickup_longitude/strided_slice�
1dense_features_1/pickup_longitude/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1dense_features_1/pickup_longitude/Reshape/shape/1�
/dense_features_1/pickup_longitude/Reshape/shapePack8dense_features_1/pickup_longitude/strided_slice:output:0:dense_features_1/pickup_longitude/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:21
/dense_features_1/pickup_longitude/Reshape/shape�
)dense_features_1/pickup_longitude/ReshapeReshapeinputs_pickup_longitude8dense_features_1/pickup_longitude/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2+
)dense_features_1/pickup_longitude/Reshape�
dense_features_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
dense_features_1/concat/axis�
dense_features_1/concatConcatV22dense_features_1/dropoff_latitude/Reshape:output:03dense_features_1/dropoff_longitude/Reshape:output:01dense_features_1/passenger_count/Reshape:output:01dense_features_1/pickup_latitude/Reshape:output:02dense_features_1/pickup_longitude/Reshape:output:0%dense_features_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
dense_features_1/concat�
hidden1/MatMul/ReadVariableOpReadVariableOp&hidden1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
hidden1/MatMul/ReadVariableOp�
hidden1/MatMulMatMul dense_features_1/concat:output:0%hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
hidden1/MatMul�
hidden1/BiasAdd/ReadVariableOpReadVariableOp'hidden1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
hidden1/BiasAdd/ReadVariableOp�
hidden1/BiasAddBiasAddhidden1/MatMul:product:0&hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
hidden1/BiasAddp
hidden1/ReluReluhidden1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
hidden1/Relu�
hidden2/MatMul/ReadVariableOpReadVariableOp&hidden2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
hidden2/MatMul/ReadVariableOp�
hidden2/MatMulMatMulhidden1/Relu:activations:0%hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
hidden2/MatMul�
hidden2/BiasAdd/ReadVariableOpReadVariableOp'hidden2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
hidden2/BiasAdd/ReadVariableOp�
hidden2/BiasAddBiasAddhidden2/MatMul:product:0&hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
hidden2/BiasAddp
hidden2/ReluReluhidden2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
hidden2/Relu�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
output/MatMul/ReadVariableOp�
output/MatMulMatMulhidden2/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output/MatMul�
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output/BiasAdd�
IdentityIdentityoutput/BiasAdd:output:0^hidden1/BiasAdd/ReadVariableOp^hidden1/MatMul/ReadVariableOp^hidden2/BiasAdd/ReadVariableOp^hidden2/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesy
w:���������:���������:���������:���������:���������::::::2@
hidden1/BiasAdd/ReadVariableOphidden1/BiasAdd/ReadVariableOp2>
hidden1/MatMul/ReadVariableOphidden1/MatMul/ReadVariableOp2@
hidden2/BiasAdd/ReadVariableOphidden2/BiasAdd/ReadVariableOp2>
hidden2/MatMul/ReadVariableOphidden2/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:7 3
1
_user_specified_nameinputs/dropoff_latitude:84
2
_user_specified_nameinputs/dropoff_longitude:62
0
_user_specified_nameinputs/passenger_count:62
0
_user_specified_nameinputs/pickup_latitude:73
1
_user_specified_nameinputs/pickup_longitude
�
�
"__inference_signature_wrapper_5543
dropoff_latitude
dropoff_longitude
passenger_count
pickup_latitude
pickup_longitude"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldropoff_latitudedropoff_longitudepassenger_countpickup_latitudepickup_longitudestatefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__wrapped_model_52952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesy
w:���������:���������:���������:���������:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:0 ,
*
_user_specified_namedropoff_latitude:1-
+
_user_specified_namedropoff_longitude:/+
)
_user_specified_namepassenger_count:/+
)
_user_specified_namepickup_latitude:0,
*
_user_specified_namepickup_longitude
�
�
@__inference_output_layer_call_and_return_conditional_losses_5421

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
)__inference_sequential_layer_call_fn_5713
inputs_dropoff_latitude
inputs_dropoff_longitude
inputs_passenger_count
inputs_pickup_latitude
inputs_pickup_longitude"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_dropoff_latitudeinputs_dropoff_longitudeinputs_passenger_countinputs_pickup_latitudeinputs_pickup_longitudestatefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_55102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesy
w:���������:���������:���������:���������:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:7 3
1
_user_specified_nameinputs/dropoff_latitude:84
2
_user_specified_nameinputs/dropoff_longitude:62
0
_user_specified_nameinputs/passenger_count:62
0
_user_specified_nameinputs/pickup_latitude:73
1
_user_specified_nameinputs/pickup_longitude
�
�
D__inference_sequential_layer_call_and_return_conditional_losses_5477

inputs
inputs_1
inputs_2
inputs_3
inputs_4*
&hidden1_statefulpartitionedcall_args_1*
&hidden1_statefulpartitionedcall_args_2*
&hidden2_statefulpartitionedcall_args_1*
&hidden2_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity��hidden1/StatefulPartitionedCall�hidden2/StatefulPartitionedCall�output/StatefulPartitionedCall�
 dense_features_1/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_dense_features_1_layer_call_and_return_conditional_losses_53532"
 dense_features_1/PartitionedCall�
hidden1/StatefulPartitionedCallStatefulPartitionedCall)dense_features_1/PartitionedCall:output:0&hidden1_statefulpartitionedcall_args_1&hidden1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:��������� **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_hidden1_layer_call_and_return_conditional_losses_53762!
hidden1/StatefulPartitionedCall�
hidden2/StatefulPartitionedCallStatefulPartitionedCall(hidden1/StatefulPartitionedCall:output:0&hidden2_statefulpartitionedcall_args_1&hidden2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_hidden2_layer_call_and_return_conditional_losses_53992!
hidden2/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall(hidden2/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_54212 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^hidden1/StatefulPartitionedCall ^hidden2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesy
w:���������:���������:���������:���������:���������::::::2B
hidden1/StatefulPartitionedCallhidden1/StatefulPartitionedCall2B
hidden2/StatefulPartitionedCallhidden2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�j
�
 __inference__traced_restore_6015
file_prefix.
*assignvariableop_sequential_hidden1_kernel.
*assignvariableop_1_sequential_hidden1_bias0
,assignvariableop_2_sequential_hidden2_kernel.
*assignvariableop_3_sequential_hidden2_bias/
+assignvariableop_4_sequential_output_kernel-
)assignvariableop_5_sequential_output_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count8
4assignvariableop_13_adam_sequential_hidden1_kernel_m6
2assignvariableop_14_adam_sequential_hidden1_bias_m8
4assignvariableop_15_adam_sequential_hidden2_kernel_m6
2assignvariableop_16_adam_sequential_hidden2_bias_m7
3assignvariableop_17_adam_sequential_output_kernel_m5
1assignvariableop_18_adam_sequential_output_bias_m8
4assignvariableop_19_adam_sequential_hidden1_kernel_v6
2assignvariableop_20_adam_sequential_hidden1_bias_v8
4assignvariableop_21_adam_sequential_hidden2_kernel_v6
2assignvariableop_22_adam_sequential_hidden2_bias_v7
3assignvariableop_23_adam_sequential_output_kernel_v5
1assignvariableop_24_adam_sequential_output_bias_v
identity_26��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B)layer-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)layer-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)layer-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB'layer-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBClayer-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBElayer-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBElayer-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBElayer-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBClayer-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp*assignvariableop_sequential_hidden1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp*assignvariableop_1_sequential_hidden1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp,assignvariableop_2_sequential_hidden2_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp*assignvariableop_3_sequential_hidden2_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp+assignvariableop_4_sequential_output_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp)assignvariableop_5_sequential_output_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0	*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp4assignvariableop_13_adam_sequential_hidden1_kernel_mIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp2assignvariableop_14_adam_sequential_hidden1_bias_mIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp4assignvariableop_15_adam_sequential_hidden2_kernel_mIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp2assignvariableop_16_adam_sequential_hidden2_bias_mIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp3assignvariableop_17_adam_sequential_output_kernel_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp1assignvariableop_18_adam_sequential_output_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_sequential_hidden1_kernel_vIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp2assignvariableop_20_adam_sequential_hidden1_bias_vIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_sequential_hidden2_kernel_vIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp2assignvariableop_22_adam_sequential_hidden2_bias_vIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp3assignvariableop_23_adam_sequential_output_kernel_vIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp1assignvariableop_24_adam_sequential_output_bias_vIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25�
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_26"#
identity_26Identity_26:output:0*y
_input_shapesh
f: :::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
�
�
%__inference_output_layer_call_fn_5825

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_54212
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�k
�
D__inference_sequential_layer_call_and_return_conditional_losses_5683
inputs_dropoff_latitude
inputs_dropoff_longitude
inputs_passenger_count
inputs_pickup_latitude
inputs_pickup_longitude*
&hidden1_matmul_readvariableop_resource+
'hidden1_biasadd_readvariableop_resource*
&hidden2_matmul_readvariableop_resource+
'hidden2_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity��hidden1/BiasAdd/ReadVariableOp�hidden1/MatMul/ReadVariableOp�hidden2/BiasAdd/ReadVariableOp�hidden2/MatMul/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOp�
'dense_features_1/dropoff_latitude/ShapeShapeinputs_dropoff_latitude*
T0*
_output_shapes
:2)
'dense_features_1/dropoff_latitude/Shape�
5dense_features_1/dropoff_latitude/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5dense_features_1/dropoff_latitude/strided_slice/stack�
7dense_features_1/dropoff_latitude/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/dropoff_latitude/strided_slice/stack_1�
7dense_features_1/dropoff_latitude/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/dropoff_latitude/strided_slice/stack_2�
/dense_features_1/dropoff_latitude/strided_sliceStridedSlice0dense_features_1/dropoff_latitude/Shape:output:0>dense_features_1/dropoff_latitude/strided_slice/stack:output:0@dense_features_1/dropoff_latitude/strided_slice/stack_1:output:0@dense_features_1/dropoff_latitude/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/dense_features_1/dropoff_latitude/strided_slice�
1dense_features_1/dropoff_latitude/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1dense_features_1/dropoff_latitude/Reshape/shape/1�
/dense_features_1/dropoff_latitude/Reshape/shapePack8dense_features_1/dropoff_latitude/strided_slice:output:0:dense_features_1/dropoff_latitude/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:21
/dense_features_1/dropoff_latitude/Reshape/shape�
)dense_features_1/dropoff_latitude/ReshapeReshapeinputs_dropoff_latitude8dense_features_1/dropoff_latitude/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2+
)dense_features_1/dropoff_latitude/Reshape�
(dense_features_1/dropoff_longitude/ShapeShapeinputs_dropoff_longitude*
T0*
_output_shapes
:2*
(dense_features_1/dropoff_longitude/Shape�
6dense_features_1/dropoff_longitude/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6dense_features_1/dropoff_longitude/strided_slice/stack�
8dense_features_1/dropoff_longitude/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8dense_features_1/dropoff_longitude/strided_slice/stack_1�
8dense_features_1/dropoff_longitude/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8dense_features_1/dropoff_longitude/strided_slice/stack_2�
0dense_features_1/dropoff_longitude/strided_sliceStridedSlice1dense_features_1/dropoff_longitude/Shape:output:0?dense_features_1/dropoff_longitude/strided_slice/stack:output:0Adense_features_1/dropoff_longitude/strided_slice/stack_1:output:0Adense_features_1/dropoff_longitude/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0dense_features_1/dropoff_longitude/strided_slice�
2dense_features_1/dropoff_longitude/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :24
2dense_features_1/dropoff_longitude/Reshape/shape/1�
0dense_features_1/dropoff_longitude/Reshape/shapePack9dense_features_1/dropoff_longitude/strided_slice:output:0;dense_features_1/dropoff_longitude/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:22
0dense_features_1/dropoff_longitude/Reshape/shape�
*dense_features_1/dropoff_longitude/ReshapeReshapeinputs_dropoff_longitude9dense_features_1/dropoff_longitude/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2,
*dense_features_1/dropoff_longitude/Reshape�
&dense_features_1/passenger_count/ShapeShapeinputs_passenger_count*
T0*
_output_shapes
:2(
&dense_features_1/passenger_count/Shape�
4dense_features_1/passenger_count/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4dense_features_1/passenger_count/strided_slice/stack�
6dense_features_1/passenger_count/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6dense_features_1/passenger_count/strided_slice/stack_1�
6dense_features_1/passenger_count/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6dense_features_1/passenger_count/strided_slice/stack_2�
.dense_features_1/passenger_count/strided_sliceStridedSlice/dense_features_1/passenger_count/Shape:output:0=dense_features_1/passenger_count/strided_slice/stack:output:0?dense_features_1/passenger_count/strided_slice/stack_1:output:0?dense_features_1/passenger_count/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.dense_features_1/passenger_count/strided_slice�
0dense_features_1/passenger_count/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0dense_features_1/passenger_count/Reshape/shape/1�
.dense_features_1/passenger_count/Reshape/shapePack7dense_features_1/passenger_count/strided_slice:output:09dense_features_1/passenger_count/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:20
.dense_features_1/passenger_count/Reshape/shape�
(dense_features_1/passenger_count/ReshapeReshapeinputs_passenger_count7dense_features_1/passenger_count/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2*
(dense_features_1/passenger_count/Reshape�
&dense_features_1/pickup_latitude/ShapeShapeinputs_pickup_latitude*
T0*
_output_shapes
:2(
&dense_features_1/pickup_latitude/Shape�
4dense_features_1/pickup_latitude/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4dense_features_1/pickup_latitude/strided_slice/stack�
6dense_features_1/pickup_latitude/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6dense_features_1/pickup_latitude/strided_slice/stack_1�
6dense_features_1/pickup_latitude/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6dense_features_1/pickup_latitude/strided_slice/stack_2�
.dense_features_1/pickup_latitude/strided_sliceStridedSlice/dense_features_1/pickup_latitude/Shape:output:0=dense_features_1/pickup_latitude/strided_slice/stack:output:0?dense_features_1/pickup_latitude/strided_slice/stack_1:output:0?dense_features_1/pickup_latitude/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.dense_features_1/pickup_latitude/strided_slice�
0dense_features_1/pickup_latitude/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0dense_features_1/pickup_latitude/Reshape/shape/1�
.dense_features_1/pickup_latitude/Reshape/shapePack7dense_features_1/pickup_latitude/strided_slice:output:09dense_features_1/pickup_latitude/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:20
.dense_features_1/pickup_latitude/Reshape/shape�
(dense_features_1/pickup_latitude/ReshapeReshapeinputs_pickup_latitude7dense_features_1/pickup_latitude/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2*
(dense_features_1/pickup_latitude/Reshape�
'dense_features_1/pickup_longitude/ShapeShapeinputs_pickup_longitude*
T0*
_output_shapes
:2)
'dense_features_1/pickup_longitude/Shape�
5dense_features_1/pickup_longitude/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5dense_features_1/pickup_longitude/strided_slice/stack�
7dense_features_1/pickup_longitude/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/pickup_longitude/strided_slice/stack_1�
7dense_features_1/pickup_longitude/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7dense_features_1/pickup_longitude/strided_slice/stack_2�
/dense_features_1/pickup_longitude/strided_sliceStridedSlice0dense_features_1/pickup_longitude/Shape:output:0>dense_features_1/pickup_longitude/strided_slice/stack:output:0@dense_features_1/pickup_longitude/strided_slice/stack_1:output:0@dense_features_1/pickup_longitude/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/dense_features_1/pickup_longitude/strided_slice�
1dense_features_1/pickup_longitude/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1dense_features_1/pickup_longitude/Reshape/shape/1�
/dense_features_1/pickup_longitude/Reshape/shapePack8dense_features_1/pickup_longitude/strided_slice:output:0:dense_features_1/pickup_longitude/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:21
/dense_features_1/pickup_longitude/Reshape/shape�
)dense_features_1/pickup_longitude/ReshapeReshapeinputs_pickup_longitude8dense_features_1/pickup_longitude/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2+
)dense_features_1/pickup_longitude/Reshape�
dense_features_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
dense_features_1/concat/axis�
dense_features_1/concatConcatV22dense_features_1/dropoff_latitude/Reshape:output:03dense_features_1/dropoff_longitude/Reshape:output:01dense_features_1/passenger_count/Reshape:output:01dense_features_1/pickup_latitude/Reshape:output:02dense_features_1/pickup_longitude/Reshape:output:0%dense_features_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
dense_features_1/concat�
hidden1/MatMul/ReadVariableOpReadVariableOp&hidden1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
hidden1/MatMul/ReadVariableOp�
hidden1/MatMulMatMul dense_features_1/concat:output:0%hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
hidden1/MatMul�
hidden1/BiasAdd/ReadVariableOpReadVariableOp'hidden1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
hidden1/BiasAdd/ReadVariableOp�
hidden1/BiasAddBiasAddhidden1/MatMul:product:0&hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
hidden1/BiasAddp
hidden1/ReluReluhidden1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
hidden1/Relu�
hidden2/MatMul/ReadVariableOpReadVariableOp&hidden2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
hidden2/MatMul/ReadVariableOp�
hidden2/MatMulMatMulhidden1/Relu:activations:0%hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
hidden2/MatMul�
hidden2/BiasAdd/ReadVariableOpReadVariableOp'hidden2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
hidden2/BiasAdd/ReadVariableOp�
hidden2/BiasAddBiasAddhidden2/MatMul:product:0&hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
hidden2/BiasAddp
hidden2/ReluReluhidden2/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
hidden2/Relu�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
output/MatMul/ReadVariableOp�
output/MatMulMatMulhidden2/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output/MatMul�
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output/BiasAdd�
IdentityIdentityoutput/BiasAdd:output:0^hidden1/BiasAdd/ReadVariableOp^hidden1/MatMul/ReadVariableOp^hidden2/BiasAdd/ReadVariableOp^hidden2/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesy
w:���������:���������:���������:���������:���������::::::2@
hidden1/BiasAdd/ReadVariableOphidden1/BiasAdd/ReadVariableOp2>
hidden1/MatMul/ReadVariableOphidden1/MatMul/ReadVariableOp2@
hidden2/BiasAdd/ReadVariableOphidden2/BiasAdd/ReadVariableOp2>
hidden2/MatMul/ReadVariableOphidden2/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:7 3
1
_user_specified_nameinputs/dropoff_latitude:84
2
_user_specified_nameinputs/dropoff_longitude:62
0
_user_specified_nameinputs/passenger_count:62
0
_user_specified_nameinputs/pickup_latitude:73
1
_user_specified_nameinputs/pickup_longitude
�	
�
A__inference_hidden2_layer_call_and_return_conditional_losses_5399

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
D__inference_sequential_layer_call_and_return_conditional_losses_5510

inputs
inputs_1
inputs_2
inputs_3
inputs_4*
&hidden1_statefulpartitionedcall_args_1*
&hidden1_statefulpartitionedcall_args_2*
&hidden2_statefulpartitionedcall_args_1*
&hidden2_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identity��hidden1/StatefulPartitionedCall�hidden2/StatefulPartitionedCall�output/StatefulPartitionedCall�
 dense_features_1/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_dense_features_1_layer_call_and_return_conditional_losses_53532"
 dense_features_1/PartitionedCall�
hidden1/StatefulPartitionedCallStatefulPartitionedCall)dense_features_1/PartitionedCall:output:0&hidden1_statefulpartitionedcall_args_1&hidden1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:��������� **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_hidden1_layer_call_and_return_conditional_losses_53762!
hidden1/StatefulPartitionedCall�
hidden2/StatefulPartitionedCallStatefulPartitionedCall(hidden1/StatefulPartitionedCall:output:0&hidden2_statefulpartitionedcall_args_1&hidden2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_hidden2_layer_call_and_return_conditional_losses_53992!
hidden2/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall(hidden2/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_54212 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^hidden1/StatefulPartitionedCall ^hidden2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesy
w:���������:���������:���������:���������:���������::::::2B
hidden1/StatefulPartitionedCallhidden1/StatefulPartitionedCall2B
hidden2/StatefulPartitionedCallhidden2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�A
�
J__inference_dense_features_1_layer_call_and_return_conditional_losses_5763
features_dropoff_latitude
features_dropoff_longitude
features_passenger_count
features_pickup_latitude
features_pickup_longitude
identityy
dropoff_latitude/ShapeShapefeatures_dropoff_latitude*
T0*
_output_shapes
:2
dropoff_latitude/Shape�
$dropoff_latitude/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$dropoff_latitude/strided_slice/stack�
&dropoff_latitude/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&dropoff_latitude/strided_slice/stack_1�
&dropoff_latitude/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&dropoff_latitude/strided_slice/stack_2�
dropoff_latitude/strided_sliceStridedSlicedropoff_latitude/Shape:output:0-dropoff_latitude/strided_slice/stack:output:0/dropoff_latitude/strided_slice/stack_1:output:0/dropoff_latitude/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
dropoff_latitude/strided_slice�
 dropoff_latitude/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 dropoff_latitude/Reshape/shape/1�
dropoff_latitude/Reshape/shapePack'dropoff_latitude/strided_slice:output:0)dropoff_latitude/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
dropoff_latitude/Reshape/shape�
dropoff_latitude/ReshapeReshapefeatures_dropoff_latitude'dropoff_latitude/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2
dropoff_latitude/Reshape|
dropoff_longitude/ShapeShapefeatures_dropoff_longitude*
T0*
_output_shapes
:2
dropoff_longitude/Shape�
%dropoff_longitude/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%dropoff_longitude/strided_slice/stack�
'dropoff_longitude/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'dropoff_longitude/strided_slice/stack_1�
'dropoff_longitude/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'dropoff_longitude/strided_slice/stack_2�
dropoff_longitude/strided_sliceStridedSlice dropoff_longitude/Shape:output:0.dropoff_longitude/strided_slice/stack:output:00dropoff_longitude/strided_slice/stack_1:output:00dropoff_longitude/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
dropoff_longitude/strided_slice�
!dropoff_longitude/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!dropoff_longitude/Reshape/shape/1�
dropoff_longitude/Reshape/shapePack(dropoff_longitude/strided_slice:output:0*dropoff_longitude/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2!
dropoff_longitude/Reshape/shape�
dropoff_longitude/ReshapeReshapefeatures_dropoff_longitude(dropoff_longitude/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2
dropoff_longitude/Reshapev
passenger_count/ShapeShapefeatures_passenger_count*
T0*
_output_shapes
:2
passenger_count/Shape�
#passenger_count/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#passenger_count/strided_slice/stack�
%passenger_count/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%passenger_count/strided_slice/stack_1�
%passenger_count/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%passenger_count/strided_slice/stack_2�
passenger_count/strided_sliceStridedSlicepassenger_count/Shape:output:0,passenger_count/strided_slice/stack:output:0.passenger_count/strided_slice/stack_1:output:0.passenger_count/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
passenger_count/strided_slice�
passenger_count/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
passenger_count/Reshape/shape/1�
passenger_count/Reshape/shapePack&passenger_count/strided_slice:output:0(passenger_count/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
passenger_count/Reshape/shape�
passenger_count/ReshapeReshapefeatures_passenger_count&passenger_count/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2
passenger_count/Reshapev
pickup_latitude/ShapeShapefeatures_pickup_latitude*
T0*
_output_shapes
:2
pickup_latitude/Shape�
#pickup_latitude/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#pickup_latitude/strided_slice/stack�
%pickup_latitude/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%pickup_latitude/strided_slice/stack_1�
%pickup_latitude/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%pickup_latitude/strided_slice/stack_2�
pickup_latitude/strided_sliceStridedSlicepickup_latitude/Shape:output:0,pickup_latitude/strided_slice/stack:output:0.pickup_latitude/strided_slice/stack_1:output:0.pickup_latitude/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
pickup_latitude/strided_slice�
pickup_latitude/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
pickup_latitude/Reshape/shape/1�
pickup_latitude/Reshape/shapePack&pickup_latitude/strided_slice:output:0(pickup_latitude/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
pickup_latitude/Reshape/shape�
pickup_latitude/ReshapeReshapefeatures_pickup_latitude&pickup_latitude/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2
pickup_latitude/Reshapey
pickup_longitude/ShapeShapefeatures_pickup_longitude*
T0*
_output_shapes
:2
pickup_longitude/Shape�
$pickup_longitude/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$pickup_longitude/strided_slice/stack�
&pickup_longitude/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&pickup_longitude/strided_slice/stack_1�
&pickup_longitude/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&pickup_longitude/strided_slice/stack_2�
pickup_longitude/strided_sliceStridedSlicepickup_longitude/Shape:output:0-pickup_longitude/strided_slice/stack:output:0/pickup_longitude/strided_slice/stack_1:output:0/pickup_longitude/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
pickup_longitude/strided_slice�
 pickup_longitude/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 pickup_longitude/Reshape/shape/1�
pickup_longitude/Reshape/shapePack'pickup_longitude/strided_slice:output:0)pickup_longitude/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2 
pickup_longitude/Reshape/shape�
pickup_longitude/ReshapeReshapefeatures_pickup_longitude'pickup_longitude/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2
pickup_longitude/Reshapee
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat/axis�
concatConcatV2!dropoff_latitude/Reshape:output:0"dropoff_longitude/Reshape:output:0 passenger_count/Reshape:output:0 pickup_latitude/Reshape:output:0!pickup_longitude/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*r
_input_shapesa
_:���������:���������:���������:���������:���������:9 5
3
_user_specified_namefeatures/dropoff_latitude::6
4
_user_specified_namefeatures/dropoff_longitude:84
2
_user_specified_namefeatures/passenger_count:84
2
_user_specified_namefeatures/pickup_latitude:95
3
_user_specified_namefeatures/pickup_longitude
�	
�
A__inference_hidden2_layer_call_and_return_conditional_losses_5801

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
)__inference_sequential_layer_call_fn_5519
dropoff_latitude
dropoff_longitude
passenger_count
pickup_latitude
pickup_longitude"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldropoff_latitudedropoff_longitudepassenger_countpickup_latitudepickup_longitudestatefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_55102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapesy
w:���������:���������:���������:���������:���������::::::22
StatefulPartitionedCallStatefulPartitionedCall:0 ,
*
_user_specified_namedropoff_latitude:1-
+
_user_specified_namedropoff_longitude:/+
)
_user_specified_namepassenger_count:/+
)
_user_specified_namepickup_latitude:0,
*
_user_specified_namepickup_longitude
�	
�
A__inference_hidden1_layer_call_and_return_conditional_losses_5376

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
&__inference_hidden1_layer_call_fn_5790

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:��������� **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_hidden1_layer_call_and_return_conditional_losses_53762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
@__inference_output_layer_call_and_return_conditional_losses_5818

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
M
dropoff_latitude9
"serving_default_dropoff_latitude:0���������
O
dropoff_longitude:
#serving_default_dropoff_longitude:0���������
K
passenger_count8
!serving_default_passenger_count:0���������
K
pickup_latitude8
!serving_default_pickup_latitude:0���������
M
pickup_longitude9
"serving_default_pickup_longitude:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�*
layer-0
layer-1
layer-2
layer-3
	optimizer
_training_endpoints
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
*U&call_and_return_all_conditional_losses
V__call__
W_default_save_signature"�(
_tf_keras_sequential�'{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "DenseFeatures", "config": {"name": "dense_features_1", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "NumericColumn", "config": {"key": "dropoff_latitude", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "dropoff_longitude", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "passenger_count", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "pickup_latitude", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "pickup_longitude", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}]}}, {"class_name": "Dense", "config": {"name": "hidden1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden2", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": [null]}, "is_graph_network": false, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "DenseFeatures", "config": {"name": "dense_features_1", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "NumericColumn", "config": {"key": "dropoff_latitude", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "dropoff_longitude", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "passenger_count", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "pickup_latitude", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "pickup_longitude", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}]}}, {"class_name": "Dense", "config": {"name": "hidden1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden2", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": [null]}}, "training_config": {"loss": "mse", "metrics": ["rmse"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�	
_feature_columns

_resources
	variables
regularization_losses
trainable_variables
	keras_api
*X&call_and_return_all_conditional_losses
Y__call__"�
_tf_keras_layer�{"class_name": "DenseFeatures", "name": "dense_features_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_features_1", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "NumericColumn", "config": {"key": "dropoff_latitude", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "dropoff_longitude", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "passenger_count", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "pickup_latitude", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "pickup_longitude", "shape": [1], "default_value": null, "dtype": "float32", "normalizer_fn": null}}]}, "_is_feature_layer": true}
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*Z&call_and_return_all_conditional_losses
[__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "hidden1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}}
�

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*\&call_and_return_all_conditional_losses
]__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "hidden2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden2", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}}
�

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
*^&call_and_return_all_conditional_losses
___call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}}
�
$iter

%beta_1

&beta_2
	'decay
(learning_ratemImJmKmLmMmNvOvPvQvRvSvT"
	optimizer
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
�
	variables

)layers
regularization_losses
*layer_regularization_losses
	trainable_variables
+non_trainable_variables
,metrics
V__call__
W_default_save_signature
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
,
`serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

-layers
	variables
.layer_regularization_losses
regularization_losses
trainable_variables
/non_trainable_variables
0metrics
Y__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
+:) 2sequential/hidden1/kernel
%:# 2sequential/hidden1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

1layers
	variables
2layer_regularization_losses
regularization_losses
trainable_variables
3non_trainable_variables
4metrics
[__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
+:) 2sequential/hidden2/kernel
%:#2sequential/hidden2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

5layers
	variables
6layer_regularization_losses
regularization_losses
trainable_variables
7non_trainable_variables
8metrics
]__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
*:(2sequential/output/kernel
$:"2sequential/output/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

9layers
 	variables
:layer_regularization_losses
!regularization_losses
"trainable_variables
;non_trainable_variables
<metrics
___call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
=0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	>total
	?count
@
_fn_kwargs
A	variables
Bregularization_losses
Ctrainable_variables
D	keras_api
*a&call_and_return_all_conditional_losses
b__call__"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "rmse", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "rmse", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

Elayers
A	variables
Flayer_regularization_losses
Bregularization_losses
Ctrainable_variables
Gnon_trainable_variables
Hmetrics
b__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0:. 2 Adam/sequential/hidden1/kernel/m
*:( 2Adam/sequential/hidden1/bias/m
0:. 2 Adam/sequential/hidden2/kernel/m
*:(2Adam/sequential/hidden2/bias/m
/:-2Adam/sequential/output/kernel/m
):'2Adam/sequential/output/bias/m
0:. 2 Adam/sequential/hidden1/kernel/v
*:( 2Adam/sequential/hidden1/bias/v
0:. 2 Adam/sequential/hidden2/kernel/v
*:(2Adam/sequential/hidden2/bias/v
/:-2Adam/sequential/output/kernel/v
):'2Adam/sequential/output/bias/v
�2�
D__inference_sequential_layer_call_and_return_conditional_losses_5683
D__inference_sequential_layer_call_and_return_conditional_losses_5434
D__inference_sequential_layer_call_and_return_conditional_losses_5452
D__inference_sequential_layer_call_and_return_conditional_losses_5613�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_sequential_layer_call_fn_5713
)__inference_sequential_layer_call_fn_5486
)__inference_sequential_layer_call_fn_5519
)__inference_sequential_layer_call_fn_5698�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference__wrapped_model_5295�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *���
���
>
dropoff_latitude*�'
dropoff_latitude���������
@
dropoff_longitude+�(
dropoff_longitude���������
<
passenger_count)�&
passenger_count���������
<
pickup_latitude)�&
pickup_latitude���������
>
pickup_longitude*�'
pickup_longitude���������
�2�
J__inference_dense_features_1_layer_call_and_return_conditional_losses_5763�
���
FullArgSpec9
args1�.
jself

jfeatures
jcols_to_output_tensors
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_dense_features_1_layer_call_fn_5772�
���
FullArgSpec9
args1�.
jself

jfeatures
jcols_to_output_tensors
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_hidden1_layer_call_and_return_conditional_losses_5783�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_hidden1_layer_call_fn_5790�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_hidden2_layer_call_and_return_conditional_losses_5801�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_hidden2_layer_call_fn_5808�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_output_layer_call_and_return_conditional_losses_5818�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_output_layer_call_fn_5825�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
B}
"__inference_signature_wrapper_5543dropoff_latitudedropoff_longitudepassenger_countpickup_latitudepickup_longitude
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
__inference__wrapped_model_5295����
���
���
>
dropoff_latitude*�'
dropoff_latitude���������
@
dropoff_longitude+�(
dropoff_longitude���������
<
passenger_count)�&
passenger_count���������
<
pickup_latitude)�&
pickup_latitude���������
>
pickup_longitude*�'
pickup_longitude���������
� "3�0
.
output_1"�
output_1����������
J__inference_dense_features_1_layer_call_and_return_conditional_losses_5763����
���
���
G
dropoff_latitude3�0
features/dropoff_latitude���������
I
dropoff_longitude4�1
features/dropoff_longitude���������
E
passenger_count2�/
features/passenger_count���������
E
pickup_latitude2�/
features/pickup_latitude���������
G
pickup_longitude3�0
features/pickup_longitude���������

 
� "%�"
�
0���������
� �
/__inference_dense_features_1_layer_call_fn_5772����
���
���
G
dropoff_latitude3�0
features/dropoff_latitude���������
I
dropoff_longitude4�1
features/dropoff_longitude���������
E
passenger_count2�/
features/passenger_count���������
E
pickup_latitude2�/
features/pickup_latitude���������
G
pickup_longitude3�0
features/pickup_longitude���������

 
� "�����������
A__inference_hidden1_layer_call_and_return_conditional_losses_5783\/�,
%�"
 �
inputs���������
� "%�"
�
0��������� 
� y
&__inference_hidden1_layer_call_fn_5790O/�,
%�"
 �
inputs���������
� "���������� �
A__inference_hidden2_layer_call_and_return_conditional_losses_5801\/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� y
&__inference_hidden2_layer_call_fn_5808O/�,
%�"
 �
inputs��������� 
� "�����������
@__inference_output_layer_call_and_return_conditional_losses_5818\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� x
%__inference_output_layer_call_fn_5825O/�,
%�"
 �
inputs���������
� "�����������
D__inference_sequential_layer_call_and_return_conditional_losses_5434����
���
���
>
dropoff_latitude*�'
dropoff_latitude���������
@
dropoff_longitude+�(
dropoff_longitude���������
<
passenger_count)�&
passenger_count���������
<
pickup_latitude)�&
pickup_latitude���������
>
pickup_longitude*�'
pickup_longitude���������
p

 
� "%�"
�
0���������
� �
D__inference_sequential_layer_call_and_return_conditional_losses_5452����
���
���
>
dropoff_latitude*�'
dropoff_latitude���������
@
dropoff_longitude+�(
dropoff_longitude���������
<
passenger_count)�&
passenger_count���������
<
pickup_latitude)�&
pickup_latitude���������
>
pickup_longitude*�'
pickup_longitude���������
p 

 
� "%�"
�
0���������
� �
D__inference_sequential_layer_call_and_return_conditional_losses_5613����
���
���
E
dropoff_latitude1�.
inputs/dropoff_latitude���������
G
dropoff_longitude2�/
inputs/dropoff_longitude���������
C
passenger_count0�-
inputs/passenger_count���������
C
pickup_latitude0�-
inputs/pickup_latitude���������
E
pickup_longitude1�.
inputs/pickup_longitude���������
p

 
� "%�"
�
0���������
� �
D__inference_sequential_layer_call_and_return_conditional_losses_5683����
���
���
E
dropoff_latitude1�.
inputs/dropoff_latitude���������
G
dropoff_longitude2�/
inputs/dropoff_longitude���������
C
passenger_count0�-
inputs/passenger_count���������
C
pickup_latitude0�-
inputs/pickup_latitude���������
E
pickup_longitude1�.
inputs/pickup_longitude���������
p 

 
� "%�"
�
0���������
� �
)__inference_sequential_layer_call_fn_5486����
���
���
>
dropoff_latitude*�'
dropoff_latitude���������
@
dropoff_longitude+�(
dropoff_longitude���������
<
passenger_count)�&
passenger_count���������
<
pickup_latitude)�&
pickup_latitude���������
>
pickup_longitude*�'
pickup_longitude���������
p

 
� "�����������
)__inference_sequential_layer_call_fn_5519����
���
���
>
dropoff_latitude*�'
dropoff_latitude���������
@
dropoff_longitude+�(
dropoff_longitude���������
<
passenger_count)�&
passenger_count���������
<
pickup_latitude)�&
pickup_latitude���������
>
pickup_longitude*�'
pickup_longitude���������
p 

 
� "�����������
)__inference_sequential_layer_call_fn_5698����
���
���
E
dropoff_latitude1�.
inputs/dropoff_latitude���������
G
dropoff_longitude2�/
inputs/dropoff_longitude���������
C
passenger_count0�-
inputs/passenger_count���������
C
pickup_latitude0�-
inputs/pickup_latitude���������
E
pickup_longitude1�.
inputs/pickup_longitude���������
p

 
� "�����������
)__inference_sequential_layer_call_fn_5713����
���
���
E
dropoff_latitude1�.
inputs/dropoff_latitude���������
G
dropoff_longitude2�/
inputs/dropoff_longitude���������
C
passenger_count0�-
inputs/passenger_count���������
C
pickup_latitude0�-
inputs/pickup_latitude���������
E
pickup_longitude1�.
inputs/pickup_longitude���������
p 

 
� "�����������
"__inference_signature_wrapper_5543����
� 
���
>
dropoff_latitude*�'
dropoff_latitude���������
@
dropoff_longitude+�(
dropoff_longitude���������
<
passenger_count)�&
passenger_count���������
<
pickup_latitude)�&
pickup_latitude���������
>
pickup_longitude*�'
pickup_longitude���������"3�0
.
output_1"�
output_1���������