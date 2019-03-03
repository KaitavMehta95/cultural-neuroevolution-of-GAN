import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import os
import importlib
import random
import time
import torch
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import math
from copy import deepcopy


###################################################################################################
######### Auxiliary Functions
###################################################################################################
def incerement_imgCount():
    global image_count
    image_count = image_count +1# This function set the seeds of the tensorflow function
# to make this notebook's output stable across runs

def xavier_init(shape):
    in_dim = shape[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=shape, stddev=xavier_stddev)

# custom weights initialization called on netG and netD
#normal
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#xavier
def uniform(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.uniform_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

#xavier
def ones(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.ones_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.ones_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)




def incerement_ganCount():
    global GanCount
    GanCount = GanCount + 1


#########################################  DEAP FUNCTIONS ##########################################################

#####################################################################################################

 #Make generators randomly first i.e select all parameters! For simplicity keep dimensions fixed
def init_individual(ind_class):
    #init activation functions with generators
    netG = Generator(ngpu,1).to(device)

    #Apply weight initaliation
    # initWeight = random.randint(0, 1)
    # if(initWeight == 0):
    #     netG.apply(weights_init)
    # elif(initWeight == 1):
    #     netG.apply(weights_init)

    netG.apply(weights_init)
        # netG.apply(uniform)
    fmeasure = divergence_measures[np.random.randint(len(divergence_measures))]

    print(fmeasure)
    gan_no = GanCount
    iter = batch_iteration[np.random.randint(len(batch_iteration))]
    iter = 1
    print (iter)

    gloops = gen_loop[np.random.randint(len(gen_loop))]
    gloops =1
    print (gloops)
    incerement_ganCount()

    lr = 0.001

    #Make discriminator for each individual
    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)


    my_gan_descriptor = GAN_Descriptor(fmeasure, netG, netD,gan_no, lr, iter, gloops)

    ind = ind_class(my_gan_descriptor)

    return ind



#  Generator on basis of loss function.
def eval_gan_approx_igd(individual):
    global All_Evals
    global All_Evals
    global best_val
    global Eval_Records
    # Get the loss function of the individual and generate loss after training and return the loss function of generator
    # In minimax Generator try to reduce the loss of itself, while increaing the loss of discriminator

    start_time = time.time()
    # GAN Descriptor has data about generaters hyper parameters and generator itself
    my_gan_descriptor = individual.gan_descriptor

    my_gan = GAN(my_gan_descriptor)
    my_gan.training_definition()
    gLoss = my_gan.separated_running(mb_size, my_gan_descriptor.batch_iteration, my_gan_descriptor.gen_loop, currentGen)
    elapsed_time = time.time() - start_time

    return gLoss, elapsed_time
##########################################################################


#
# Crossover will be backpropogating gradient loss of two different generators. - Not working
   # Gradient based crossover for the generators net paramenters, ofspring produced by
    # Back propogating through indv1 generated samples loss and vice versa.
def cxGAN(ind1, ind2):

    descriptor1 = ind1.gan_descriptor
    descriptor2= ind2.gan_descriptor

    descriptor1.netD = deepcopy(descriptor2.netD)

    """Crossover between two GANs
       The offspring is produced by mating of the two GANs 
    """

    return ind1, ind2


#####################################################################################################
# Add different type of mutations
def mutGAN(individual):
    """Different types of mutations for the GAN.
       Only of the networks is mutated (Discriminator or Generator)
       Each time a network is mutated, only one mutation operator is applied
    """

    my_gan_descriptor = individual.gan_descriptor
    type_mutation = mutation_types[np.random.randint(len(mutation_types))]
    #
    if type_mutation == "learning_rate":
        initial_lrate = 0.1
        k = 0.1
        lrate = individual.gan_descriptor.lr
        lrate = initial_lrate * math.exp(-k * currentGen)

        if(lrate < 0.0000):
            lrate = 0.001
        individual.gan_descriptor.lr = lrate
    if type_mutation == "batch_iteration":
        individual.gan_descriptor.iter = batch_iteration[np.random.randint(len(batch_iteration))]
    if type_mutation == "gloops":
        individual.gan_descriptor.gloops = gen_loop[np.random.randint(len(gen_loop))]
    # if type_mutation == "fmeasure":
    #     individual.gan_descriptor.fmeasure = divergence_measures[np.random.randint(len(divergence_measures))]

    return individual,

#crossover of weights which is not working!! Produces grey images.
# def crossOver(gen1,gen2):
    # print("______CROSS OVER_____________")
    # indv1 = GAN(gen1)
    # indv2 = GAN(gen2)
    # indv1.training_definition()
    # indv2.training_definition()
    #
    # for i, data in enumerate(dataloader, 0):
    #
    #     ############################
    #     # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    #     ###########################
    #     ## Train with all-real batch
    #     netD.zero_grad()
    #     # Format batch
    #     real_cpu = data[0].to(device)
    #     b_size = real_cpu.size(0)
    #     label = torch.full((b_size,), real_label, device=device)
    #     # Forward pass real batch through D
    #     output = netD(real_cpu).view(-1)
    #     # Calculate loss on all-real batch
    #     errD_real = indv1.criterion(output, label)
    #     # Calculate gradients for D in backward pass
    #     errD_real.backward()
    #     D_x = output.mean().item()
    #
    #     ## Train with all-fake batch
    #     # Generate batch of latent vectors
    #     noise = torch.randn(b_size, nz, 1, 1, device=device)
    #     # Generate fake image batch with G Individual 1
    #     fake = indv1.descriptor.netG(noise)
    #     label.fill_(fake_label)
    #     # Classify all fake batch with D
    #     output = netD(fake.detach()).view(-1)
    #     # Calculate D's loss on the all-fake batch
    #     errD_fake = indv1.criterion(output, label)
    #     # Calculate the gradients for this batch
    #     errD_fake.backward()
    #     D_G_z1 = output.mean().item()
    #     # Add the gradients from the all-real and all-fake batches
    #     errD = errD_real + errD_fake
    #     # Update D
    #     optimizerD.step()
    #
    #     ############################
    #     # (2) Update G Indiviual 2 network
    #     ###########################
    #     indv2.descriptor.netG.zero_grad()
    #     label.fill_(real_label)  # fake labels are real for generator cost
    #     # Since we just updated D, perform another forward pass of all-fake batch through D
    #     output = netD(fake).view(-1)
    #     # Calculate G2 loss based on this output which was generated from G1.
    #     errG = indv2.criterion(output, label)
    #     # Calculate gradients for G
    #     errG.backward()
    #     D_G_z2 = output.mean().item()
    #
    #     # Update G- Individual 2
    #     indv2.optimizerG.step()
    #     i = i+ random.randint(2,4)
    # return indv2.descriptor.netG

###############################################################################################################################

class MyContainer(object):
    # This class does not require the fitness attribute
    # it will be  added later by the creator
    def __init__(self, thegan_descriptor):
        # Some initialisation with received values
        self.gan_descriptor = thegan_descriptor

########################################################### GAN Descriptor  #######################################################################################################################################################################################

class GAN_Descriptor:
    def __init__(self, fmeasure="Standard_Divergence",generator='',discriminator='',gan_no=0,lr=0.001,batch_iteration=1,gen_loop = 1):
        #get generator and fmeasure
        self.netG = generator
        self.netD = discriminator
        self.fmeasure = fmeasure
        self.gan_no = gan_no
        self.lr = lr
        self.batch_iteration = batch_iteration
        self.gen_loop = gen_loop

##################################################################################################
############################################# GAN  ################################################
###################################################################################################


class GAN:
    def __init__(self, gan_descriptor):
        self.descriptor = gan_descriptor

        if(self.descriptor.fmeasure == 'BCE'):
            self.criterion = nn.BCELoss()
        elif(self.descriptor.fmeasure == 'L1Loss'):
            self.criterion = nn.L1Loss()
        elif (self.descriptor.fmeasure == 'MSELoss'):
            self.criterion = nn.MSELoss()

    def training_definition(self):
        # =============================== TRAINING ====================================

        self.optimizerG = optim.Adam(self.descriptor.netG.parameters(), lr= self.descriptor.lr, betas=(beta1, 0.999))
        self.optimizerD = optim.Adam( self.descriptor.netD.parameters(), lr= self.descriptor.lr, betas=(beta1, 0.999))


    def separated_running(self, b_size,batch_iteration,gen_loop,current_gen):



        avg_Gloss = 0
        training_iteration = 0


        #gen_loop- loop till generator training
        if(batch_iteration == 1):
            #if all samples are taken into consideration then we skip the generator loop count
            gen_loop = 1

        for count in range(gen_loop):
            for i, data in enumerate(dataloader, 0):
                # if(i == 0):
                #     print (self.descriptor.netD)
                #     print (self.descriptor.netG)

                #Define discriminator of Individual and train it
                netD =  self.descriptor.netD
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                netD.zero_grad()
                # Format batch
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, device=device)
                # Forward pass real batch through D
                output = netD(real_cpu).view(-1)
                # Calculate loss on all-real batch

                errD_real = self.criterion(output, label)
                # if(i == 0):
                #     print ("-------------Discriminator-")
                #     print (self.criterion)
                #     print (output)
                #     print ("-------------GEnerator-")
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = self.descriptor.netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.descriptor.netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()


                # Update G
                self.optimizerG.step()
                fitLoss = fitness(output, label)
                avg_Gloss = avg_Gloss + fitLoss.item()

                # Output training stats
                if training_iteration % 10 == 0:

                    temp_avg_loss = avg_Gloss/(training_iteration+1)
                    print("Generator-"+str(self.descriptor.gan_no)+"  training_iteration :"+str(training_iteration)+" loss:"+str(temp_avg_loss))
                    print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % ( i, len(dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                if training_iteration % 50 == 0:
                    # save images after every 50 sample batches.
                    fake = self.descriptor.netG(fixed_noise).detach().cpu()
                    horizontal_grid = vutils.make_grid(fake, padding=2, normalize=True)
                    # Plot and save horizontal
                    plt.imshow(np.transpose(horizontal_grid.numpy(), (1, 2, 0)))
                    # plt.show()
                    fig = plt.gcf()
                    plt.draw()

                    path = 'C:/Users/RACHIT/Desktop/Kaitav Project/FaceGenerationGAN/Output/gan'+str(self.descriptor.gan_no)+'/'
                    file_name = 'Gan'+str(self.descriptor.gan_no) +'__'+ str(image_count)+'.png'
                    saveAt = path+file_name
                    incerement_imgCount()
                    fig.savefig(saveAt, dpi=100)
                    plt.axis('off')
                    plt.close()
                    print("fig saved")
                # count var to maintain no of iterations
                training_iteration = training_iteration + 1

                #change value of i accroding to batch-size
                if(batch_iteration > 1):
                    i = i + batch_iteration
                # Save Losses for plotting later
                # G_losses.append(errG.item())
                # D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                # if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                #     with torch.no_grad():
                #         fake = netG(fixed_noise).detach().cpu()
                #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        GenLoss = avg_Gloss / len(dataloader)
        print("Generator-"+str(self.descriptor.gan_no)+" Iteration :"+ str(training_iteration) + " loss :"+str(GenLoss))
        #add the loss of generators to the file according to the generations
        gnum = self.descriptor.gan_no
        lossfile = 'gen' + str(gnum) + '.txt'
        f = open(lossfile, "a+")
        writeToFile = "Generation: {0}, Generator no : {1}, Fitness:  {2} \n".format(current_gen, gnum, GenLoss)
        f.write(writeToFile)

        if (current_gen == 49 or current_gen == 99 ):
            print("writing to file")
            # store the generator architecture at last generation
            writeToFile1 = "Fmeasure: {0}, Generator no : {1}, Batch _Itertation:  {2}, Generator Training  Loop: {3}, Learning Rate: {4} \n".format(self.descriptor.fmeasure,self.descriptor.gan_no, batch_iteration, gen_loop,self.descriptor.lr)
            f.write(writeToFile1)

        f.close()
        return GenLoss



###############################################################################################################################
########################################################### Network Descriptor #######################################################################################################################################################################################

# Generator Code

class Generator(nn.Module):
        def __init__(self, ngpu,activation_init):
            super(Generator, self).__init__()
            self.ngpu = ngpu

            if(activation_init == 0):
                self.main = nn.Sequential(
                    # input is Z, going into a convolution
                    nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 8),
                    nn.ReLU(True),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf),
                    nn.ReLU(True),
                    # state size. (ngf) x 32 x 32
                    nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                    nn.Tanh()
                    # state size. (nc) x 64 x 64
                )
            elif(activation_init == 1 ):
                self.main = nn.Sequential(
                    # input is Z, going into a convolution
                    nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 8),
                    nn.ELU(True),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 4),
                    nn.ELU(True),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 2),
                    nn.ELU(True),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf),
                    nn.ELU(True),
                    # state size. (ngf) x 32 x 32
                    nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                    nn.Tanh()
                    # state size. (nc) x 64 x 64
                )

        def forward(self, input):
            return self.main(input)

    #########################################################################
# Discriminator Code

class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)






#####################################################################################################

def Init_GA_():
    """
                         Definition of GA operators
    """
    print("Initalizing GA")

    # Minimization of the IGD measure

    creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))

    creator.create("Individual", MyContainer, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    # Structure initializers

    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_gan_approx_igd)
    toolbox.register("mate", cxGAN)
    toolbox.register("mutate", mutGAN)

    if SEL == 0:
        toolbox.register("select", tools.selBest)

    return toolbox


#####################################################################################################

def Apply_GA_GAN(toolbox, pop_size=10, gen_number=50, CXPB=0.7, MUTPB=0.3):
    """
          Application of the Genetic Algorithm
    """

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    # stats = tools.Statistics()
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    g = 0

    while g < gen_number:
        global currentGen
        currentGen = g
        print("#########################Generation: ", g)
        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Extracting all the fitnesses of
        fits = [ind.fitness.values[0] for ind in pop]
        g = g + 1

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        # invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # fitnesses = map(toolbox.evaluate, invalid_ind)
        # for ind, fit in zip(invalid_ind, fitnesses):
        #     ind.fitness.values = fit

        pop[:] = offspring


#####################################################################################################

# def main():


#####################################################################################################


if __name__ == "__main__":  # Example python3 GAN_Descriptor_Deap.py 0 1000 10 1 30 10 5 50 10 2 0 10 0

    ngpu = 1    #No of GPU utilization
    num_workers = 4 # Number of workers to load images
    npop =1
    ngen = 100
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    ###########CONSTANTS##########
    nc = 1  # Channel
    nz = 100  # noise
    ngf = 64  # Generator output
    ndf = 64  # Discriminator Input

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    fitness = nn.BCELoss()
    global image_count
    image_count = 0
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root='C:/Users/RACHIT/Desktop/Kaitav Project/FaceGenerationGAN/DCGAN-data/',
                               transform=transforms.Compose([
                                   transforms.Grayscale(1),
                                   transforms.ToTensor()

                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                             shuffle=True, num_workers=4)

    # The    parameters of the program are set or read from command line

    global currentGen  # Current generation
    global All_Evals
    global best_val
    global Eval_Records
    global GanCount

    myseed = 999  # Seed: Used to set different outcomes of the stochastic program

    SEL = 0
    CXp = 0.2  # Crossover probability (Mutation is 1-CXp)
    nselpop = 5  # Selected population size

    All_Evals = 0  # Tracks the number of evaluations
    best_val = 10000.0  # Tracks best value among solutions

    # the no of a generator to keep track of it
    GanCount = 1


    # List of weight initialization functions the networks can use
    # init_functions = [xavier_init, tf.random_uniform, tf.random_normal]

    #  List of latent distributions VAE can use
    # lat_functions = [np.random.uniform, np.random.normal]

    #  List of divergence measures
    divergence_measures = ["BCE"]
    # divergence_measures = ["MSELoss"]
    # Mutation types
    mutation_types = ["gloops", "fmeasure", "learning_rate", "batch_iteration"]
    batch_iteration = [1, 2, 3, 4]
    gen_loop = [1, 2, 3]

    mb_size = 128  # Minibatch size
    # number_epochs = 1001                                      # Number epochs for training
    # print_cycle = 1001                                        # Frequency information is printed
    # lr = 1e-3                                                 # Learning rate for Adam optimizer


    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0


    # GA initialization
    toolbox = Init_GA_()

    # Runs the GA
    print("Applying GA algorithm")

    Apply_GA_GAN(toolbox, pop_size=npop,
                 gen_number=ngen, CXPB=CXp, MUTPB=1 - CXp)
    print("Completed!! Congratulations.")
    # Examples of how to call the function
    # ./GAN_Descriptor_Deap.py 111 1000 10 1 30 10 5 50 20 1000 0 20 10 5