ĐT
ÎŁ
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
dtypetype
ž
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.22v2.3.1-38-g9edbe5075f78F
d
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0

NoOpNoOp
´
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*p
valuegBe B_

	split

signatures
><
VARIABLE_VALUEVariable split/.ATTRIBUTES/VARIABLE_VALUE
 

serving_default_data_inPlaceholder*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
dtype0*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ŕ
StatefulPartitionedCallStatefulPartitionedCallserving_default_data_inVariable*
Tin
2*
Tout
2*
_collective_manager_ids
 *L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_signature_wrapper_41
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ś
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *$
fR
__inference__traced_save_68

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_restore_81Ú9
Ę
j
__inference___call___27
data_in
mul_readvariableop_resource

identity_1

identity_2>
SizeSizedata_in*
T0*
_output_shapes
: 2
SizeS
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast|
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
: *
dtype02
mul/ReadVariableOpX
mulMulCast:y:0mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
mulQ
Cast_1Castmul:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1B
Size_1Sizedata_in*
T0*
_output_shapes
: 2
Size_1R
IdentityIdentitySize_1:output:0*
T0*
_output_shapes
: 2

Identityt
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackp
strided_slice/stack_1Pack
Cast_1:y:0*
N*
T0*
_output_shapes
:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ď
strided_sliceStridedSlicedata_instrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*

begin_mask2
strided_slicep
strided_slice_1/stackPack
Cast_1:y:0*
N*
T0*
_output_shapes
:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2÷
strided_slice_1StridedSlicedata_instrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
end_mask2
strided_slice_1w

Identity_1Identitystrided_slice:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity_1y

Identity_2Identitystrided_slice_1:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity_2"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::Y U
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
!
_user_specified_name	data_in
â

v
__inference__traced_restore_81
file_prefix
assignvariableop_variable

identity_2˘AssignVariableOpĆ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*S
valueJBHB split/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B 2
RestoreV2/shape_and_slicesľ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes

::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp9
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp{

Identity_1Identityfile_prefix^AssignVariableOp^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_1m

Identity_2IdentityIdentity_1:output:0^AssignVariableOp*
T0*
_output_shapes
: 2

Identity_2"!

identity_2Identity_2:output:0*
_input_shapes
: :2$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

w
 __inference_signature_wrapper_41
data_in
unknown
identity

identity_1˘StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCalldata_inunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *L
_output_shapes:
8:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 * 
fR
__inference___call___272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
!
_user_specified_name	data_in


__inference__traced_save_68
file_prefix'
#savev2_variable_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_211a38034f454a2f87fdf658d24e8a0b/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameŔ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*S
valueJBHB split/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B 2
SaveV2/shape_and_slicesŕ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĄ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: "¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_defaultđ
D
data_in9
serving_default_data_in:0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙E
output_09
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙E
output_19
StatefulPartitionedCall:1˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:š
G
	split

signatures
__call__"
_generic_user_object
: 2Variable
,
serving_default"
signature_map
ć2ă
__inference___call___27Ç
˛
FullArgSpec
args
jself
	jdata_in
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *&˘#
!˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
/B-
 __inference_signature_wrapper_41data_inŤ
__inference___call___279˘6
/˘,
*'
data_in˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "OL
$!
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
$!
1˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ĺ
 __inference_signature_wrapper_41ŔD˘A
˘ 
:Ş7
5
data_in*'
data_in˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"uŞr
7
output_0+(
output_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
7
output_1+(
output_1˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙