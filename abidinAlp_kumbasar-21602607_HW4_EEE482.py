from scipy.io import loadmat
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF
import sys

# # Question 1

# In[2]:

question = sys.argv[1]

def alp_kumbasar_21602607_HW4(question):
    if question == '1':
        with h5py.File('hw4_data1.mat', 'r') as f:
            faces = np.array(list(f['faces']))


        # In[3]:


        print(faces.shape)


        # In[4]:


        faces = faces.T


        # In[5]:


        faces.shape


        # In[6]:


        plt.imshow(faces[0].reshape(32,32).T)
        plt.show()

        # ## Question 1 A

        # In[7]:
        print("Question 1 A is running...")

        pca_100 = PCA(n_components=100, whiten = True)
        faces_pca_100 = pca_100.fit(faces)


        # In[8]:


        var_pca = faces_pca_100.explained_variance_ratio_


        # In[9]:


        print(var_pca.shape)


        # In[10]:


        plt.plot(np.arange(100), var_pca)
        plt.grid(color='black', linestyle='-', linewidth=0.2)
        plt.title("PCA Variance Graph form Most Significant to Least")
        plt.xlabel("PC Number from most significant in 0 and least in 99")
        plt.ylabel("Variance of the PC")
        plt.show()


        # In[11]:


        faces_pca_100.components_[0]


        # In[12]:


        def dispImArray(x, y, faces):
            cmap = 'gray'
            
            plt.figure(figsize=(10,10))
            for i in range(x*y):
                plt.subplot(x, y , i + 1)
                plt.imshow(faces[i].reshape((32, 32)).T, cmap=cmap)
            plt.show()


        # In[13]:


        
        dispImArray(5, 5, faces_pca_100.components_)


        # ## Question 1 B

        # In[14]:

        print("Question 1 B is running...")
        
        def loss_cal_PCA_b(original, const):
            
            loss = (original - const)**2
            
            mean = np.mean(loss)
            mean_loss = np.mean(loss, axis = 1)
            std = np.std(mean_loss)
            return mean, std


        # In[15]:


        # (x - meanPca) Com CompT + meanPca

        mean_pca = faces_pca_100.mean_

        f_10_pca = (faces - mean_pca).dot(faces_pca_100.components_[:10].T).dot(faces_pca_100.components_[:10]) + mean_pca
        f_25_pca = (faces - mean_pca).dot(faces_pca_100.components_[:25].T).dot(faces_pca_100.components_[:25]) + mean_pca
        f_50_pca = (faces - mean_pca).dot(faces_pca_100.components_[:50].T).dot(faces_pca_100.components_[:50]) + mean_pca


        # In[16]:


        
        print("First 36 Original Images")
        dispImArray(6, 6, faces)


        # In[17]:


        
        print("First 10 EigenValues Constructed Images")
        dispImArray(6, 6, f_10_pca)


        # In[18]:


        
        print("First 25 EigenValues Constructed Images")
        dispImArray(6, 6, f_25_pca)


        # In[19]:


        
        print("First 50 EigenValues Constructed Images")
        dispImArray(6, 6, f_50_pca)


        # In[20]:


        mean_10_pca, std_10_pca = loss_cal_PCA_b(faces, f_10_pca)
        mean_25_pca, std_25_pca = loss_cal_PCA_b(faces, f_25_pca)
        mean_50_pca, std_50_pca = loss_cal_PCA_b(faces, f_50_pca)


        # In[21]:


        print("Mean loss of reconstruction images using first 10 eigenvalues: " + str(mean_10_pca) +
              "\nSTD loss of reconstruction images using first 10 eigenvalues:  " + str(std_10_pca) + "\n")
        print("Mean loss of reconstruction images using first 25 eigenvalues: " + str(mean_25_pca) +
              "\nSTD loss of reconstruction images using first 25 eigenvalues:  " + str(std_25_pca)+ "\n")
        print("Mean loss of reconstruction images using first 50 eigenvalues: " + str(mean_50_pca) +
              "\nSTD loss of reconstruction images using first 50 eigenvalues:  " + str(std_50_pca)+ "\n")


        # ## Question 1 C

        # In[22]:
        
        print("Question 1 C is running...")
        
        ica_10 = FastICA(n_components=10, whiten=True, random_state=np.random.seed(5))
        ica_25 = FastICA(n_components=25, whiten=True, random_state=np.random.seed(5))
        ica_50 = FastICA(n_components=50, whiten=True, random_state=np.random.seed(5))

        ica_10 = ica_10.fit(faces)
        ica_25 = ica_25.fit(faces)
        ica_50 = ica_50.fit(faces)


        # In[23]:


        print("ICA obtained by 10 ICs")
        dispImArray(2,5,ica_10.components_)
        print("ICA obtained by 25 ICs")
        dispImArray(5,5,ica_25.components_)
        print("ICA obtained by 50 ICs")
        dispImArray(10, 5, ica_50.components_)


        # In[24]:


        sig_10 = ica_10.fit_transform(faces)
        a_10 = ica_10.mixing_
        sig_25 = ica_25.fit_transform(faces)
        a_25 = ica_25.mixing_
        sig_50 = ica_50.fit_transform(faces)
        a_50 = ica_50.mixing_


        recog_faces_10_ica = sig_10.dot(a_10.T) + ica_10.mean_
        recog_faces_25_ica = sig_25.dot(a_25.T) + ica_25.mean_
        recog_faces_50_ica = sig_50.dot(a_50.T) + ica_50.mean_


        # In[25]:


        
        print("First 25 Images obtained by first 10 ICs")
        dispImArray(5, 5, recog_faces_10_ica)


        # In[26]:


        
        print("First 25 Images obtained by first 25 ICs")
        dispImArray(5, 5, recog_faces_25_ica)


        # In[27]:


        
        print("First 25 Images obtained by first 50 ICs")
        dispImArray(5, 5, recog_faces_50_ica)


        # In[28]:


        mean_10_ica, std_10_ica = loss_cal_PCA_b(faces, recog_faces_10_ica)
        mean_25_ica, std_25_ica = loss_cal_PCA_b(faces, recog_faces_25_ica)
        mean_50_ica, std_50_ica = loss_cal_PCA_b(faces, recog_faces_50_ica)


        # In[29]:


        print("Mean loss of reconstruction images using first 10 ICs: " + str(mean_10_ica) +
              "\nSTD loss of reconstruction images using first 10 ICs:  " + str(std_10_ica) + "\n")
        print("Mean loss of reconstruction images using first 25 ICs: " + str(mean_25_ica) +
              "\nSTD loss of reconstruction images using first 25 ICs:  " + str(std_25_ica)+ "\n")
        print("Mean loss of reconstruction images using first 50 ICs: " + str(mean_50_ica) +
              "\nSTD loss of reconstruction images using first 50 ICs:  " + str(std_50_ica)+ "\n")


        # ## Question 1 D

        # In[30]:
        
        print("Question 1 D is running...")

        abs_min = np.abs(np.min(faces))
        faces_nmf = faces + abs_min

        nmf10 = NMF(n_components=10, max_iter=500)
        nmf25 = NMF(n_components=25, max_iter=500)
        nmf50 = NMF(n_components=50, max_iter=500)

        fit_10 = nmf10.fit_transform(faces_nmf)
        fit_25 = nmf25.fit_transform(faces_nmf)
        fit_50 = nmf50.fit_transform(faces_nmf)

        comp10 = nmf10.components_
        comp25 = nmf25.components_
        comp50 = nmf50.components_


        # In[31]:


        print("NMF obtained by 10 MFs")
        dispImArray(1,5,comp10)
        print("NMF obtained by 25 MFs")
        dispImArray(5,5,comp25)
        print("NMF obtained by 50 MFs")
        dispImArray(10,5,comp50)


        # In[32]:


        faces_10_nmf = fit_10.dot(comp10) - abs_min
        faces_25_nmf = fit_25.dot(comp25) - abs_min
        faces_50_nmf = fit_50.dot(comp50) - abs_min


        # In[33]:


        
        print("First 25 Images obtained by first 10 MFs")
        dispImArray(5, 5, faces_10_nmf)


        # In[34]:


        
        print("First 25 Images obtained by first 25 MFs")
        dispImArray(5, 5, faces_25_nmf)


        # In[35]:


        
        print("First 25 Images obtained by first 50 MFs")
        dispImArray(5, 5, faces_50_nmf)


        # In[36]:


        mean_10_nmf, std_10_nmf = loss_cal_PCA_b(faces, faces_10_nmf)
        mean_25_nmf, std_25_nmf = loss_cal_PCA_b(faces, faces_25_nmf)
        mean_50_nmf, std_50_nmf = loss_cal_PCA_b(faces, faces_50_nmf)


        # In[37]:


        print("Mean loss of reconstruction images using first 10 MFs: " + str(mean_10_nmf) +
              "\nSTD loss of reconstruction images using first 10 MFs:  " + str(std_10_nmf) + "\n")
        print("Mean loss of reconstruction images using first 25 MFs: " + str(mean_25_nmf) +
              "\nSTD loss of reconstruction images using first 25 MFs:  " + str(std_25_nmf)+ "\n")
        print("Mean loss of reconstruction images using first 50 MFs: " + str(mean_50_nmf) +
              "\nSTD loss of reconstruction images using first 50 MFs:  " + str(std_50_nmf)+ "\n")


        # # Question 2

        # ## Question 2 A

        # In[38]:
    elif question == '2':
        
        print("Question 2 is running...")
        def tunning_create(a, x, m, std):
            
            ret = a * np.exp(-(x - m)**2/ (2 * std**2))
            
            return ret


        # In[39]:


        A = 1
        mu = np.arange(-10,11)
        std = 1
        x_sti = np.linspace(-15,16,1000)

        collector_2a = []

        for i in mu:
            collector_2a.append(tunning_create(A, x_sti, i, std))


        # In[40]:


        tunning_2a = np.array(collector_2a)
        print(tunning_2a.shape)

        print("Question 2 A is running...")
        # In[41]:


        for i in range(tunning_2a.shape[0]):
            plt.plot(np.linspace(-15,16,1000), tunning_2a[i])
            
        plt.title("Population of Tunning Curves for 21 Stimulus")
        plt.xlabel("Stimulus Number")
        plt.ylabel("Activity")
        plt.show()


        # In[42]:


        mean_tunning = np.zeros((1000,1))
        for i in range(tunning_2a.shape[1]):
            mean_tunning[i] = np.mean(tunning_2a[:,i])


        # In[43]:


        plt.plot( np.arange(1000), mean_tunning)
        plt.show()


        # In[44]:


        mu = np.arange(-10,11)
        res_2a_1 = tunning_create(A, -1, mu, std)


        # In[45]:


        plt.plot(np.arange(-10,11) , res_2a_1)
        plt.title("Activiy of Population x = -1 vs. Preferred Stimulus")
        plt.xlabel("Preffered Stimulus")
        plt.ylabel("Activiy of Population x = -1")
        plt.grid(color='black', linestyle='-', linewidth=0.2)
        plt.show()


        # ## Question 2 B

        # In[46]:

        print("Question 2 B is running...")

        def winner_take(stimulu, resp):
            
            ret = stimulu[np.argmax(resp)]
            
            return ret


        # In[47]:


        resp = []
        est_wta = []
        wta_error = []
        stim = []
        TRIAL = 200
        stims = np.linspace(-5,5,750)

        for i in range(TRIAL):
            stim_selec = np.random.choice(stims)
            er = np.random.normal(0, std/20, 21)
            response = tunning_create(A, stim_selec, mu, std) + er
            
            win = winner_take(mu, response)
            wta_error.append(np.abs(win - stim_selec))
            
            resp.append(response)
            stim.append(stim_selec)
            est_wta.append(win)
            


        # In[48]:


        plt.scatter(np.arange(200), est_wta)
        plt.scatter(np.arange(200), stim)
        plt.legend(["Est_WTA", "Stimulus"], loc=2)
        plt.xlabel("Number of Trial")
        plt.ylabel("Stimulus Result")
        plt.title("Actual vs. Estimated by WTA Stimulus in 200 Trial")
        plt.show()


        # In[49]:


        mean_wta_er = np.mean(np.array(wta_error))
        std_wta_er = np.std(np.array(wta_error))


        # In[50]:


        print("Mean of Error of estimation done with WTA: " + str(mean_wta_er) +
              "\nSTD of Error of estimation done with WTA: " + str(std_wta_er) + "\n")


        # ## Question 2 C

        # In[51]:

        print("Question 2 C is running...")
        
        def loglik(resp, mu, std, A, x):
            
            summer = 0
            
            for i, j in zip(resp, mu):
                summer += (i - tunning_create(A, x, j, std)) **2

            return summer


        # In[52]:


        def MLE(A, mu, std, resp, sti):
            
            log_col = []
            
            for i in sti:
                hold = (loglik(resp, mu, std, A, i))
                #print(hold.shape)
                log_col.append(hold)
            min_ = np.argmin(log_col)
            est = sti[min_]

            return est


        # In[53]:


        resp = np.array(resp)
        stim = np.array(stim)
        est_sti_coll = []
        error_2c = []
        mu = np.arange(-10,11)

        for i in range(stim.shape[0]):
            hold = MLE(A, mu, std, resp[i], np.linspace(-5,5,750))
            error_mle = np.abs(stim[i] - hold)
            
            est_sti_coll.append(hold)
            error_2c.append(error_mle)


        # In[54]:


        np.array(est_sti_coll).shape


        # In[55]:


        plt.scatter(np.arange(200), est_sti_coll, linewidths=1)
        plt.scatter(np.arange(200), stim, linewidths=0.1)
        plt.legend(["Est_MLE", "Stimulus"], loc=2)
        plt.xlabel("Number of Trial")
        plt.ylabel("Stimulus Result")
        plt.title("Actual vs. Estimated by MLE Stimulus in 200 Trial")
        plt.show()


        # In[56]:


        mean_mle_er = np.mean(np.array(error_2c))
        std_mle_er = np.std(np.array(error_2c))


        # In[57]:


        print("Mean of Error of estimation done with MLD: " + str(mean_mle_er) +
              "\nSTD of Error of estimation done with MLD: " + str(std_mle_er) + "\n")


        # ## Question 2 D

        # In[58]:
        
        print("Question 2 D is running...")

        def poster(resp, mu, std, A, x):
            
                
            summer = 0
            
            for i, j in zip(resp, mu):
                summer += (i - tunning_create(A, x, j, std)) **2
            
            summer = (summer / (2 * (std / 20)**2)) + (x**2) / (2 * 2.5 **2)
            
            return summer


        # In[59]:


        def MAP(A, mu, std, resp, sti):
            
            log_col = []
            
            for i in sti:
                hold = (poster(resp, mu, std, A, i))
                #print(hold.shape)
                log_col.append(hold)
            min_ = np.argmin(log_col)
            est = sti[min_]

            return est


        # In[60]:


        est_sti_coll_map = []
        error_2d = []


        for i in range(stim.shape[0]):
            hold = MAP(A, mu, std, resp[i], np.linspace(-5,5,750))
            error_mle = np.abs(stim[i] - hold)
            
            est_sti_coll_map.append(hold)
            error_2d.append(error_mle)


        # In[61]:


        plt.scatter(np.arange(200), est_sti_coll, linewidths=1)
        plt.scatter(np.arange(200), stim, linewidths=0.1)
        plt.legend(["Est_MAP", "Stimulus"], loc=2)
        plt.xlabel("Number of Trial")
        plt.ylabel("Stimulus Result")
        plt.title("Actual vs. Estimated by MAP Stimulus in 200 Trial")
        plt.show()


        # In[62]:


        mean_map_er = np.mean(np.array(error_2d))
        std_map_er = np.std(np.array(error_2d))


        # In[63]:


        print("Mean of Error of estimation done with MPD: " + str(mean_map_er) +
              "\nSTD of Error of estimation done with MPD: " + str(std_map_er) + "\n")


        # ## Question 2 E

        # In[64]:
        
        print("Question 2 E is running...")
        print("This question 2 e takes a while to compute. Please Wait...")

        error_2e = []
        sigmas = [0.1, 0.2, 0.5, 1, 2, 5]

        for i in range(TRIAL):
            error_holder = []
            stimu = np.random.choice(np.linspace(-5,5,1000))
            for i in sigmas:
                resp = tunning_create(A, stimu, mu, i)
                noise = np.random.normal(0, 1/20, 21)
                
                resp = resp + noise
                
                est_mle = MLE(A, mu, std, resp, np.linspace(-5,5,1000))
                hold = np.abs(stimu - est_mle)
                error_holder.append(hold)
                
            error_2e.append(error_holder)


        # In[65]:


        error_2e = np.array(error_2e)
        #print(error_2e.shape)


        # In[66]:


        mean_err_2e = []
        std_err_2e = []

        for i in range(sigmas.__len__()):
            m_hold = np.mean(error_2e[:,i])
            s_hold = np.std(error_2e[:,i])
            
            mean_err_2e.append(m_hold)
            std_err_2e.append(s_hold)
            print("For sigma value = " + str(sigmas[i]) + " mean and std of error is as follows: \n")
            print("Mean of Error of estimation done with MLE: " + str(m_hold) +
              "\nSTD of Error of estimation done with WTA: " + str(s_hold) + "\n")


        # In[67]:


        np.array(mean_err_2e).shape
        np.array(std_err_2e).shape


        # In[68]:


        plt.figure()
        plt.errorbar(np.arange(6), np.array(mean_err_2e), yerr=2*np.array(std_err_2e), ecolor = 'r', elinewidth=0.5, capsize=2)
        plt.xlabel("Mean absolute error")
        plt.ylabel("STD of tunning function")
        plt.title("MEAN ERROR OF MLE vs. STD of Tunning Function")
        plt.show()
    
    else:
        
        print("You have entered wrong number for question input. Please try again...")


alp_kumbasar_21602607_HW4(question)
