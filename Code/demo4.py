#http://blog.csdn.net/sparkexpert/article/details/70147409

# 导入需要的包
import tensorflow as tf# 编写神经网络
from tensorflow.examples.tutorials.mnist import input_data # 第一次下载数据时用
import numpy as np# 矩阵运算操作
#from skimage.io import imsave# 保存影像
import scipy
import os# 读取路径下文件
import shutil# 递归删除文件
# 图像的size为(28, 28, 1)
img_height = 28
img_width = 28
img_size = img_height * img_width

to_train = True
to_restore = False# 是否存储训练结果
output_path = "output"# 存储文件的路径

# 总迭代次数500
max_epoch = 500

h1_size = 150           #第一隐藏层的size，即特征数
h2_size = 300           #第二隐藏层的size，即特征数
z_size = 100            #生成器的传入参数
batch_size = 256


#这个实验中使用的是mnist图像数据，先定义一下图像的长宽、是否训练、是否保存、保存模型地址、最大批次
# 以及隐藏层的参数和生成器输入的维度和batch_size
'''
    函数功能：生成影像，参与训练过程
    输入：z_prior,       #输入tf格式，size为（batch_size, z_size）的数据
    输出：x_generate,    #生成图像
         g_params,      #生成图像的所有参数
    '''
# generate (model 1)

def build_generator(z_prior):
    # 第一个链接层
    # 以2倍标准差stddev的截断的正态分布中生成大小为[z_size, h1_size]的随机值，权值weight初始化。
    # 生成大小为[h1_size]的0值矩阵，偏置bias初始化
    # 通过矩阵运算，将输入z_prior传入隐含层h1。激活函数为relu
    w1 = tf.Variable(tf.truncated_normal([z_size, h1_size], stddev=0.1), name="g_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h1_size]), name="g_b1", dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(z_prior, w1) + b1)
    # 第二个链接层
    # 以2倍标准差stddev的截断的正态分布中生成大小为[h1_size, h2_size]的随机值，权值weight初始化。
    # 生成大小为[h2_size]的0值矩阵，偏置bias初始化
    # 通过矩阵运算，将h1传入隐含层h2。激活函数为relu
    w2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.1), name="g_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h2_size]), name="g_b2", dtype=tf.float32)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    # 第三个链接层
    # 以2倍标准差stddev的截断的正态分布中生成大小为[h2_size, image_size]的随机值，权值weight初始化。
    # 生成大小为[image_size]的0值矩阵，偏置bias初始化
    # 通过矩阵运算，将h2传入隐含层h3。
    w3 = tf.Variable(tf.truncated_normal([h2_size, img_size], stddev=0.1), name="g_w3", dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([img_size]), name="g_b3", dtype=tf.float32)
    h3 = tf.matmul(h2, w3) + b3
    # 利用tanh激活函数，将h3传入输出层
    x_generate = tf.nn.tanh(h3)
    g_params = [w1, b1, w2, b2, w3, b3]
    # 将所有参数合并到一起
    return x_generate, g_params

#build_generator定义了生成器，输入参数为一个长度为100的先验向量，然后通过三个全连接层
# （其实这里可以使用任何形式不一定是全连接层，变种中有ＣＮＮ和ＬＳＴＭ的）映射到长度为784的向量也就是图像扁平化后的长度，返回了映射中用到的参数
'''
    函数功能：对输入数据进行判断，并保存其参数
    输入：x_data,        #输入的真实数据
        x_generated,     #生成器生成的虚假数据
        keep_prob，      #dropout率，防止过拟合
    输出：y_data,        #判别器对batch个数据的处理结果
        y_generated,     #判别器对余下数据的处理结果
        d_params，       #判别器的参数
    '''
# discriminator (model 2)
def build_discriminator(x_data, x_generated, keep_prob):
    # tf.concat
    # 合并输入数据，包括真实数据x_data和生成器生成的假数据x_generated
    x_in = tf.concat([x_data, x_generated], 0)
    # 第一个链接层
    # 以2倍标准差stddev的截断的正态分布中生成大小为[image_size, h2_size]的随机值，权值weight初始化。
    # 生成大小为[h2_size]的0值矩阵，偏置bias初始化
    # 通过矩阵运算，将输入x_in传入隐含层h1.同时以一定的dropout率舍弃节点，防止过拟合
    w1 = tf.Variable(tf.truncated_normal([img_size, h2_size], stddev=0.1), name="d_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h2_size]), name="d_b1", dtype=tf.float32)
    h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in, w1) + b1), keep_prob)
    # 第二个链接层
    # 以2倍标准差stddev的截断的正态分布中生成大小为[h2_size, h1_size]的随机值，权值weight初始化。
    # 生成大小为[h1_size]的0值矩阵，偏置bias初始化
    # 通过矩阵运算，将h1传入隐含层h2.同时以一定的dropout率舍弃节点，防止过拟合
    w2 = tf.Variable(tf.truncated_normal([h2_size, h1_size], stddev=0.1), name="d_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h1_size]), name="d_b2", dtype=tf.float32)
    h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)
    # 第三个链接层
    # 以2倍标准差stddev的截断的正态分布中生成大小为[h1_size, 1]的随机值，权值weight初始化。
    # 生成0值，偏置bias初始化
    # 通过矩阵运算，将h2传入隐含层h3
    w3 = tf.Variable(tf.truncated_normal([h1_size, 1], stddev=0.1), name="d_w3", dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([1]), name="d_b3", dtype=tf.float32)
    h3 = tf.matmul(h2, w3) + b3
    # 从h3中切出batch_size张图像
    y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [batch_size, -1], name=None))
    # 从h3中切除余下的图像
    y_generated = tf.nn.sigmoid(tf.slice(h3, [batch_size, 0], [-1, -1], name=None))
    # 判别器的所有参数
    d_params = [w1, b1, w2, b2, w3, b3]
    return y_data, y_generated, d_params

#build_discriminator定义了区分器，每次接受一个批次的真实数据与生成数据，在全连接层后使用sigmoid计算每个数据是真实数据的概率
'''
    函数功能：输入相关参数，将运行结果以图片的形式保存到当前路径下
    输入：batch_res,       #输入数据
        fname,             #输入路径
        grid_size=(8, 8),  #默认输出图像为8*8张
        grid_pad=5，       #默认图像的边缘留白为5像素
    输出：无
    '''
#
def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], img_height, img_width)) + 0.5
    # 将batch_res进行值[0, 1]归一化，同时将其reshape成（batch_size, image_height, image_width）
    # 重构显示图像格网的参数
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    #imsave(fname, img_grid)
    #img.save('output/num.jpg')
    scipy.misc.imsave(fname, img_grid)

#train函数中首先载入数据，然后定义了占位符，接下来定义了交替训练的损失函数，g_loss是生成器的损失函数，计算的就是生产数据的交叉熵
# d_loss是整个数据的交叉熵，因为分类器要保证在所有数据上都能很好的区分，所以损失函数中包含了所有数据。
# 之后做了一些保存的文件夹操作，接着从0-1均匀分布中抽取了ｚ（至于为什么用这个分布，可以去查看一个概率论，几乎所有重要的概率分布都可以从均匀分布Uniform(0,1)中生成出来）
# 接着就是交替训练以及生产一个训练好的生成器生成的图片了。
'''
    函数功能：训练整个GAN网络，并随机生成手写数字
    输入：无
    输出：sess.saver()
    '''
def train():
    # load data（mnist手写数据集）
    mnist = input_data.read_data_sets('mnist_data', one_hot=True)
    # 定义GAN网络的输入，其中x_data为[batch_size, image_size], z_prior为[batch_size, z_size]
    x_data = tf.placeholder(tf.float32, [batch_size, img_size], name="x_data")
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
    # 定义dropout率
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # 创建生成模型
    # 利用生成器生成数据x_generated和参数g_params
    x_generated, g_params = build_generator(z_prior)
    # 创建判别模型
    y_data, y_generated, d_params = build_discriminator(x_data, x_generated, keep_prob)

    # 损失函数的设置
    d_loss = - (tf.log(y_data) + tf.log(1 - y_generated))
    g_loss = - tf.log(y_generated)
    # 设置学习率为0.0001，用AdamOptimizer进行优化
    optimizer = tf.train.AdamOptimizer(0.0001)

    # 两个模型的优化函数
    # 判别器discriminator 和生成器 generator 对损失函数进行最小化处理
    d_trainer = optimizer.minimize(d_loss, var_list=d_params)
    g_trainer = optimizer.minimize(g_loss, var_list=g_params)

    # 全局变量初始化
    init = tf.initialize_all_variables()
    # 启动会话sess
    saver = tf.train.Saver()
    # 启动默认图
    sess = tf.Session()
    # 初始化
    sess.run(init)

    if to_restore:
        chkpt_fname = tf.train.latest_checkpoint(output_path)
        saver.restore(sess, chkpt_fname)
    else:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)
    # 利用随机正态分布产生噪声影像，尺寸为(batch_size, z_size)
    z_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)

    steps = 60000 / batch_size
    # 逐个epoch内训练
    for i in range(sess.run(global_step), max_epoch):
        for j in np.arange(steps):
            #         for j in range(steps):
            print("epoch:%s, iter:%s" % (i, j))
            # 每一步迭代，我们都会加载256个训练样本，然后执行一次train_step
            x_value, _ = mnist.train.next_batch(batch_size)
            x_value = 2 * x_value.astype(np.float32) - 1
            z_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)

            # 执行生成# 每个batch下，输入数据运行GAN，训练判别器
            sess.run(d_trainer,
                     feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
            # 执行判别# 每个batch下，输入数据运行GAN，训练生成器
            if j % 1 == 0:
                sess.run(g_trainer,
                         feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
        # 每一个epoch中的所有batch训练完后，利用z_sample测试训练后的生成器
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_sample_val})
        # 每一个epoch中的所有batch训练完后，显示生成器的结果，并打印生成结果的值
        show_result(x_gen_val, "output/sample{0}.jpg".format(i))
        # 每一个epoch中，生成随机分布以重置z_random_sample_val
        z_random_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
        # 每一个epoch中，利用z_random_sample_val生成手写数字图像，并显示结果
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_random_sample_val})
        show_result(x_gen_val, "output/random_sample{0}.jpg".format(i))
        sess.run(tf.assign(global_step, i + 1))
        saver.save(sess, os.path.join(output_path, "model"), global_step=global_step)


def test():
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
    x_generated, _ = build_generator(z_prior)
    chkpt_fname = tf.train.latest_checkpoint(output_path)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(init)
    saver.restore(sess, chkpt_fname)
    z_test_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
    x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_test_value})
    show_result(x_gen_val, "output/test_result.jpg")


if __name__ == '__main__':
    if to_train:
        train()
    else:
        test()