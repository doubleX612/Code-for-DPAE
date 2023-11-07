import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.models import Model,Sequential,load_model
from tensorflow.python.keras.layers import Input,BatchNormalization,Concatenate,Add,Multiply,Dropout
from tensorflow.python.keras.optimizers import Adam,SGD,RMSprop,Adadelta,Adagrad
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.python.keras.callbacks import LearningRateScheduler,ModelCheckpoint
from keras import metrics

def build_eeg_denoise_mlp(rate1,rate2,nb_input):
    
    ee_in = Input(shape=(nb_input,),name='ee_in')
 
    # dual pathway encoder
    nb_hid1 = int(nb_input*pow(rate1/100,1))
    nb_hid2 = int(nb_input/pow(rate2/100,1))
    clean_hid = Dense(units=nb_hid1,kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='selu')(ee_in)
    noise_hid = Dense(units=nb_hid2,kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='selu')(ee_in)
    nb_hid1 = int(nb_hid1*pow(rate1/100,1))
    nb_hid2 = int(nb_hid2*pow(rate2/100,1))
    clean_hid = Dense(units=nb_hid1,kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='selu')(clean_hid)
    noise_hid = Dense(units=nb_hid2,kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='selu')(noise_hid)
    nb_hid1 = int(nb_hid1*pow(rate1/100,1))
    nb_hid2 = int(nb_hid2*pow(rate2/100,1))
    clean_hid = Dense(units=nb_hid1,kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='selu')(clean_hid)
    noise_hid = Dense(units=nb_hid2,kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='selu')(noise_hid)
    con_layer =  Concatenate()([clean_hid, noise_hid])
    con_layer = BatchNormalization()(con_layer)
    
    #fusion block + skipconnect
    fus_hid = nb_hid1+nb_hid2
    fusion_block = Dense(units=fus_hid*0.45,kernel_initializer='he_uniform',
                         bias_initializer='TruncatedNormal',activation='selu')(con_layer)
    fusion_block = Dense(units=int(fus_hid*(pow(0.45,2))),kernel_initializer='he_uniform',
                         bias_initializer='TruncatedNormal',activation='selu')(fusion_block)
    fusion_block = Dense(units=int(fus_hid*(pow(0.45,3))),kernel_initializer='he_uniform',
                         bias_initializer='TruncatedNormal',activation='selu')(fusion_block)
    fusion_block = Dense(units=int(fus_hid*(pow(0.45,2))),kernel_initializer='he_uniform',
                         bias_initializer='TruncatedNormal',activation='selu')(fusion_block)
    fusion_block = Dense(units=int(fus_hid*0.45),kernel_initializer='he_uniform',
                         bias_initializer='TruncatedNormal',activation='selu')(fusion_block)

    clean_fusion = Dense(units=nb_hid1,kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='selu')(fusion_block)
    noise_fusion = Dense(units=nb_hid2,kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='selu')(fusion_block)
    
    clean_hid = Add()([clean_hid,clean_fusion])
    noise_hid = Add()([noise_hid,noise_fusion])

    # dual pathway decoder
    nb_hid1 = int(nb_hid1*pow(rate1/100,1))
    nb_hid2 = int(nb_hid2*pow(rate2/100,1))
    clean_hid = Dense(units=nb_hid1,kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='selu')(clean_hid)
    noise_hid = Dense(units=nb_hid2,kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='selu')(noise_hid)
    con_layer =  Concatenate()([clean_hid, noise_hid])
    con_layer = BatchNormalization()(con_layer)

    con_layer = Dense(units=256,kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='selu')(con_layer)
    ee_out = Dense(units=nb_input,name='ee_out',activation='selu',bias_initializer='TruncatedNormal',kernel_initializer='glorot_uniform')(con_layer)
    model=Model(inputs=ee_in,outputs=ee_out)
    model.summary()
    return model

def build_eeg_denoise_1dcnn(rate1,rate2,nb_input):
    
    ee_in = Input(shape=(nb_input,1),name='ee_in')
    clean_hid = Conv1D(16, 3, activation='selu', input_shape=(nb_input, 1), padding="valid",strides=2)(ee_in)
    noise_hid = Conv1D(16, 5, activation='selu', input_shape=(nb_input, 1), padding="valid",strides=4)(ee_in)
    
    clean_hid = Conv1D(16, 3, activation='selu',kernel_initializer='he_uniform',
                       bias_initializer='TruncatedNormal',padding="valid",strides=2)(clean_hid)
    noise_hid = Conv1D(16, 5, activation='selu',kernel_initializer='he_uniform',
                       bias_initializer='TruncatedNormal',padding="valid",strides=4)(noise_hid)
    clean_hid = Conv1D(16, 3, activation='selu',kernel_initializer='he_uniform',
                       bias_initializer='TruncatedNormal',padding="valid",strides=2)(clean_hid)
    noise_hid = Conv1D(16, 5, activation='selu',kernel_initializer='he_uniform',
                       bias_initializer='TruncatedNormal',padding="valid",strides=4)(noise_hid)
    clean_hid = Flatten()(clean_hid)
    noise_hid = Flatten()(noise_hid)
    con_layer =  Concatenate()([clean_hid, noise_hid])
    con_layer = BatchNormalization()(con_layer)
    
    #fusion block + skipconnect
    fus_hid =  con_layer.shape[1]
    fusion_block = Dense(units=int(fus_hid)*0.45,kernel_initializer='he_uniform',
                         bias_initializer='TruncatedNormal',activation='selu')(con_layer)
    fusion_block = Dense(units=int(int(fus_hid)*(pow(0.45,2))),kernel_initializer='he_uniform',
                         bias_initializer='TruncatedNormal',activation='selu')(fusion_block)
    fusion_block = Dense(units=int(int(fus_hid)*(pow(0.45,3))),kernel_initializer='he_uniform',
                         bias_initializer='TruncatedNormal',activation='selu')(fusion_block)
    fusion_block = Dense(units=int(int(fus_hid)*(pow(0.45,2))),kernel_initializer='he_uniform',
                         bias_initializer='TruncatedNormal',activation='selu')(fusion_block)
    fusion_block = Dense(units=int(int(fus_hid)*0.45),kernel_initializer='he_uniform',
                         bias_initializer='TruncatedNormal',activation='selu')(fusion_block)
#     fusion_block = BatchNormalization()(fusion_block)
    clean_fusion = Dense(units=clean_hid.shape[1],
                         kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='selu')(fusion_block)
    noise_fusion = Dense(units=noise_hid.shape[1],
                         kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='selu')(fusion_block)
    
    clean_hid = Add()([clean_hid,clean_fusion])
    noise_hid = Add()([noise_hid,noise_fusion])

    con_layer =  Concatenate()([clean_hid, noise_hid])
    con_layer = BatchNormalization()(con_layer)
    con_layer = Flatten()(con_layer)
    ee_out = Dense(units=nb_input,name='ee_out',activation='tanh',bias_initializer='TruncatedNormal',kernel_initializer='glorot_uniform')(con_layer)
    model=Model(inputs=ee_in,outputs=ee_out)
#     model.summary()
    return model

def build_eeg_denoise_1drnn(rate1,rate2,nb_input):
    
    ee_in = Input(shape=(1,nb_input),name='ee_in')
    nb_hid1 = int(nb_input*pow(rate1/100,1))
    nb_hid2 = int(nb_input*pow(rate2/100,1))
    clean_hid = GRU(units=nb_hid1,kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',
                     activation='relu',input_shape=(1,nb_input),return_sequences=True)(ee_in)
    noise_hid = GRU(units=nb_hid2,kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',
                     activation='relu',input_shape=(1,nb_input),return_sequences=True)(ee_in)
   
   
    # dualpathway
    clean_hid = Flatten()(clean_hid)
    noise_hid = Flatten()(noise_hid)

    con_layer =  Concatenate()([clean_hid, noise_hid])
    con_layer = BatchNormalization()(con_layer)
    
    #fusion block + skipconnect
    fus_hid = con_layer.shape[1]
    fusion_block = Dense(units=int(fus_hid)*0.45,kernel_initializer='he_uniform',
                         bias_initializer='TruncatedNormal',activation='relu')(con_layer)
    fusion_block = Dense(units=int(fus_hid)*(pow(0.45,2)),kernel_initializer='he_uniform',
                         bias_initializer='TruncatedNormal',activation='relu')(fusion_block)
    fusion_block = Dense(units=int(fus_hid)*(pow(0.45,3)),kernel_initializer='he_uniform',
                         bias_initializer='TruncatedNormal',activation='relu')(fusion_block)
    fusion_block = Dense(units=int(fus_hid)*(pow(0.45,2)),kernel_initializer='he_uniform',
                         bias_initializer='TruncatedNormal',activation='relu')(fusion_block)
    fusion_block = Dense(units=int(fus_hid)*0.45,kernel_initializer='he_uniform',
                         bias_initializer='TruncatedNormal',activation='relu')(fusion_block)
#     fusion_block = BatchNormalization()(fusion_block)
    clean_fusion = Dense(units=clean_hid.shape[1],
                         kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='selu')(fusion_block)
    noise_fusion = Dense(units=noise_hid.shape[1],
                         kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='selu')(fusion_block)
    
    clean_hid = Add()([clean_hid,clean_fusion])
    noise_hid = Add()([noise_hid,noise_fusion])
    
    con_layer =  Concatenate()([clean_hid, noise_hid])
    con_layer = BatchNormalization()(con_layer)

#     con_layer = Dense(units=256,kernel_initializer='he_uniform',bias_initializer='TruncatedNormal',activation='selu')(con_layer)
    ee_out = Dense(units=nb_input,name='ee_out',activation='tanh',bias_initializer='TruncatedNormal',kernel_initializer='glorot_uniform')(con_layer)
    model=Model(inputs=ee_in,outputs=ee_out)
#     model.summary()
    return model


dpae_model = build_eeg_denoise_mlp(45,75,512)
# plot_model(test_model,show_shapes='True')