from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from six.moves import xrange
from collections import namedtuple
from scipy import misc

from module import *

class semanticgan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.load_size = args.load_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir
        self.with_flip=args.flip
        self.dataset_name=self.dataset_dir.split('/')[-1]
        self.dataset_dim = args.dataset_dim
        self.num_step = args.step

        self.G = args.G
        self.DR = args.DR
        self.DF = args.DF
        self.sem_DA_fake = args.sem_DA_fake
        self.sem_DA_real = args.sem_DA_real
        self.sem_G_fake = args.sem_G_fake
        self.ssim_G = args.ssim_G
        self.sem_DA_fake_adversarial=  args.sem_DA_fake_adversarial

        self.trainA = args.trainA
        self.trainASem = args.trainASem
        self.trainB = args.trainB
        self.trainBSem = args.trainBSem
        self.testA = args.testA
        self.testASem = args.testASem
        self.testB = args.testB
        self.testBSem = args.testBSem

        self.discriminator = discriminator
        if args.use_resnet:
            self.generator = generator_resnet
        else:
            self.generator = generator_unet
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        self.criterionSem = sem_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc))
        if args.phase=='train':
            self._build_model()
            self.saver = tf.train.Saver(max_to_keep=2)

    def _build_model(self):
        immy_a,_ ,_,immy_a_sem= self.build_input_image_op(self.trainA, self.trainASem,False)
        immy_b,_ ,_,immy_b_sem= self.build_input_image_op(self.trainB, self.trainBSem,False)

        self.real_A,self.real_B, self.real_A_sem,self.real_B_sem = tf.train.shuffle_batch([immy_a,immy_b,immy_a_sem,immy_b_sem],self.batch_size,150,30,8)

        self.fake_A = self.generator(self.real_B, self.options, False, name="generatorB2A")
        self.DA_fake,self.DSEM_A_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")
        self.DA_real, self.DSEM_A_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
        
        self.dsem_loss_real =  self.criterionSem(self.DSEM_A_real,self.real_A_sem)
        self.dsem_loss_fake= self.criterionSem(self.DSEM_A_fake, self.real_B_sem)
        self.dsem_loss_fake_adversarial = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DSEM_A_fake,labels=tf.ones_like(self.DSEM_A_fake)*1/35))

        self.g_ssim_loss = ssim_criterion(self.fake_A,self.real_B)
        self.g_loss_b2a = (self.G * self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) + self.sem_G_fake * self.dsem_loss_fake + self.ssim_G* self.g_ssim_loss)/2 #+ self.L1_lambda * abs_criterion(self.real_A, self.fake_A)
        
        self.da_loss_real = self.DR * self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.DF * self.criterionGAN(self.DA_fake, tf.zeros_like(self.DA_fake)) 
        
        self.da_loss = (self.da_loss_real  + self.sem_DA_real * self.dsem_loss_real + self.da_loss_fake + self.sem_DA_fake * self.dsem_loss_fake + self.sem_DA_fake_adversarial* self.dsem_loss_fake_adversarial) / 5

        self.g_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)

        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.dsmea_loss_real_sum = tf.summary.scalar("dsema_loss_real",self.dsem_loss_real)
        
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.dsmea_loss_fake_sum = tf.summary.scalar("dsema_loss_fake",self.dsem_loss_fake)
        self.dsema_loss_fake_ad_sum = tf.summary.scalar("dsema_loss_fake_ad",self.dsem_loss_fake_adversarial)

        self.g_ssim_loss_sum =tf.summary.scalar("ssmi_g_loss",self.g_ssim_loss)

        self.da_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.dsmea_loss_real_sum,  self.da_loss_fake_sum, self.dsmea_loss_fake_sum, self.dsema_loss_fake_ad_sum,self.g_ssim_loss_sum]
        )

        immy_test_b,path_b,_,_ = self.build_input_image_op(self.testB,self.testBSem, True)

        self.test_B,self.test_path_b = tf.train.batch([immy_test_b,path_b],1,2,100)
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")
        _, self.test_B_sem = self.discriminator(self.testA,self.options, True, name = 'discriminatorA')
        
        t_vars = tf.trainable_variables()
        self.da_vars = [var for var in t_vars if 'discriminatorA' in var.name]
        self.g_vars_b2a = [var for var in t_vars if 'generatorB2A' in var.name]

    def build_input_image_op(self,dir, dirSem, is_test=False):
        def _parse_function(image_tensor):
            image = tf.read_file(image_tensor[0])
            image_sem = tf.read_file(image_tensor[1])
            image = tf.image.decode_image(image, channels = 3)
            image_sem = tf.image.decode_image( image_sem , channels = 1)
            image.set_shape([None, None, self.input_c_dim])
            image_sem.set_shape([None, None,1])
            return image , image_tensor[0], image_sem

        samples = [os.path.join(dir, s) for s in os.listdir(dir)]
        samples_sem = [s.replace(dir, dirSem) for s in samples]

        image_tensor = tf.constant(np.stack((samples, samples_sem), axis = -1))

        dataset = tf.contrib.data.Dataset.from_tensor_slices(image_tensor)
        dataset = dataset.map(_parse_function)
        num_iteration = self.num_step
        dataset = dataset.repeat(num_iteration)
        iterator = dataset.make_one_shot_iterator()
        image , image_path, image_sem = iterator.get_next()
        
        im_shape= tf.shape(image)

        #change range of value o [-1,1]
        image = tf.image.convert_image_dtype(image,tf.float32)
        image = (image*2)-1

        if not is_test:
            #resize to load_size
            image = tf.image.resize_images(image,[self.load_size//2,self.load_size])
            image_sem = tf.image.resize_images(image_sem, [self.load_size//2,self.load_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            #crop fine_size

            if(self.load_size - self.image_size != 0):
                crop_offset_w = tf.random_uniform((), minval=0, maxval=tf.shape(image)[1] - self.image_size, dtype=tf.int32)
            else:
                crop_offset_w = 0
            
            if(self.load_size//2 - self.image_size != 0):
                crop_offset_h = tf.random_uniform((), minval=0, maxval= tf.shape(image)[0]- self.image_size, dtype=tf.int32)
            else:
                crop_offset_h = 0

            image = tf.image.crop_to_bounding_box(image, crop_offset_h, crop_offset_w, self.image_size , self.image_size )          
            image_sem = tf.image.crop_to_bounding_box(image_sem, crop_offset_h, crop_offset_w, self.image_size, self.image_size)          
            #random flip left right
            # if self.with_flip:
            #     image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize_images(image,[self.load_size//2,self.load_size])
            image_sem = tf.image.resize_images(image_sem, [self.load_size//2,self.load_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            
        return image,image_path,im_shape, image_sem

    def train(self, args):
        """Train cyclegan"""
        self.da_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
            .minimize(self.da_loss, var_list=self.da_vars)
        self.g_b2a_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
            .minimize(self.g_loss_b2a, var_list=self.g_vars_b2a)

        image_summaries = []

        #summaries for training
        tf.summary.image('train_A',self.real_A)
        tf.summary.image('train_A_Sem',self.real_A_sem)
        
        tf.summary.image('train_B',self.real_B)
        tf.summary.image('train_B_Sem',self.real_B_sem)
        
        tf.summary.image('B_to_A',self.fake_A)
        
        self.pred_sem_real_image = tf.argmax(self.DSEM_A_real, dimension=3, name="prediction")
        self.pred_sem_real_image = tf.expand_dims(self.pred_sem_real_image, dim=3)

        self.pred_sem_fake_image = tf.argmax(self.DSEM_A_fake, dimension=3, name="prediction")
        self.pred_sem_fake_image = tf.expand_dims(self.pred_sem_fake_image, dim=3)

        tf.summary.image('pred_sem_A', tf.cast(self.pred_sem_real_image,tf.uint8))
        tf.summary.image('pred_sem_B', tf.cast(self.pred_sem_fake_image,tf.uint8))

        tf.summary.image('test_B',self.test_B)
        tf.summary.image('test_B_to_A',self.testA)
        
        self.test_B_sem_image = tf.argmax(self.test_B_sem, dimension=3, name="prediction")
        self.test_B_sem_image = tf.expand_dims(self.test_B_sem_image, dim=3)
     
        tf.summary.image('pred_sem_test_B',tf.cast(self.test_B_sem_image,tf.uint8))

        init_op = [tf.global_variables_initializer(),tf.local_variables_initializer()]
        
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(args.checkpoint_dir, self.sess.graph)

        summary_op = tf.summary.merge_all()

        self.counter = 0
        

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners()
        print('Thread running')

        print('Start step: {}'.format(self.counter))

        for idx in range(self.counter, self.num_step):
            self.counter += 1
            start_time = time.time()

            # Update G network + Update D network
            lossG, lossD, _,_ = self.sess.run([self.g_loss_b2a,self.da_loss, self.g_b2a_optim,self.da_optim])

            print(("Step: [%4d/%4d], LossG: %.3f, LossD: %.3f Time batch: %4.4f" \
                    % (idx, self.num_step, lossG, lossD, time.time() - start_time)))

            if np.mod(self.counter, 200) == 1:
                summary_string = self.sess.run(summary_op)
                self.writer.add_summary(summary_string,self.counter)

            if np.mod(self.counter, 1000) == 2:
                self.save(args.checkpoint_dir,self.counter)

        coord.request_stop()
        coord.join(stop_grace_period_secs=10)

    def save(self, checkpoint_dir, step):
        model_name = "semanticGAN"

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        def get_var_to_restore_list(ckpt_path, mask=[], prefix=""):
            """
            Get all the variable defined in a ckpt file and add them to the returned var_to_restore list. Allows for partially defined model to be restored fomr ckpt files.
            Args:
                ckpt_path: path to the ckpt model to be restored
                mask: list of layers to skip
                prefix: prefix string before the actual layer name in the graph definition
            """
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            variables_dict = {}
            for v in variables:
                name = v.name[:-2]
                skip=False
                #check for skip
                for m in mask:
                    if m in name:
                        skip=True
                        continue
                if not skip:
                    variables_dict[v.name[:-2]] = v
            #print(variables_dict)
            reader = tf.train.NewCheckpointReader(ckpt_path)
            var_to_shape_map = reader.get_variable_to_shape_map()
            var_to_restore = {}
            for key in var_to_shape_map:
                #print(key)
                if prefix+key in variables_dict.keys():
                    var_to_restore[key] = variables_dict[prefix+key]
            return var_to_restore

        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            q = ckpt.model_checkpoint_path.split("-")[-1]
            print("Restored step: ", q)
            self.counter= int(q) 
            savvy = tf.train.Saver(var_list=get_var_to_restore_list(ckpt.model_checkpoint_path))
            savvy.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False

    def test(self, args):
        """Test""" 
        sample_op, sample_path,im_shape,sample_op_sem = self.build_input_image_op(self.testB ,self.testBSem,is_test=True)
        sample_batch,path_batch,im_shapes,sample_sem_batch = tf.train.batch([sample_op,sample_path,im_shape,sample_op_sem],batch_size=self.batch_size,num_threads=4,capacity=self.batch_size*50,allow_smaller_final_batch=True)

        gen_images = self.generator(sample_batch,self.options,name='generatorB2A') 
        _,sem_images = self.discriminator(gen_images, self.options,name='discriminatorA')
        
        sem_images_out = tf.argmax(sem_images, dimension=3, name="prediction")
        sem_images_out = tf.cast(tf.expand_dims(sem_images_out, dim=3),tf.uint8)

        #init everything
        self.sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

        #start queue runners
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners()
        print('Thread running')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if not os.path.exists(args.test_dir): #python 2 is dumb...
            os.makedirs(args.test_dir)

        print('Starting')
        for count in range(0, args.dataset_dim//self.batch_size):
            try:
                print('Processed images: [%d/%d]' % (count,args.dataset_dim // self.batch_size), end='\n')
                pred_sem_imgs,fake_imgs,sample_images,sample_paths,im_sps, sem_gt = self.sess.run([sem_images_out,gen_images,sample_batch,path_batch,im_shapes,sample_sem_batch])
                #iterate over each sample in the batch
                for rr in range(pred_sem_imgs.shape[0]):
                    #create output destination
                    dest_path = sample_paths[rr].decode('UTF-8').replace(self.testB,args.test_dir)
                    parent_destination = os.path.abspath(os.path.join(dest_path, os.pardir))
                    if not os.path.exists(parent_destination):
                        os.makedirs(parent_destination)
                    
                    im_sp = im_sps[rr]

                    if(not args.SoG):
                        fake_img = ((fake_imgs[rr]+1)/2)*255
                        fake_img = misc.imresize(fake_img,(im_sp[0],im_sp[1]))
                        misc.imsave(dest_path,fake_img)
                    else:
                        pred_sem_img = misc.imresize(np.squeeze(pred_sem_imgs[rr],axis=-1),(im_sp[0],im_sp[1]))
                        misc.imsave(dest_path,pred_sem_img)
                count+=1
            except Exception as e:
                print(e)
                break;

        print('Elaboration complete')
        coord.request_stop()
        coord.join(stop_grace_period_secs=10)