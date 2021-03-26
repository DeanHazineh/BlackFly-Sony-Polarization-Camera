import os
import shutil
import datetime  
import PySpin
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io as sci
import pickle
import keyboard
import time
import numpy as np
import cv2 
import glob
# Use Agg backend for canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def addAxis(thisfig, n1, n2):
    axlist = []
    for i in range(n1*n2):
        axlist.append(thisfig.add_subplot(n1,n2,i+1))
    return np.array(axlist)

def addColorbar(thisfig, thisax, thisim):
    divider = make_axes_locatable(thisax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    thisfig.colorbar(thisim, cax=cax, orientation='vertical')
    return

def formatPlots(thisfig, thisax, imhandle, xlabel, ylabel, title,
    rmvxLabel=False, rmvyLabel=False, addcolorbar=True,
    setxlim=[], setylim=[]):
    thisax.set_xlabel(xlabel)
    thisax.set_ylabel(ylabel)
    thisax.set_title(title)
    if addcolorbar:
        addColorbar(thisfig, thisax, imhandle)

    if rmvxLabel:
        thisax.set_xticklabels([''])
        thisax.set_xlabel('')
    if rmvyLabel:
        thisax.set_yticklabels([''])
        thisax.set_ylabel('')
    
    if setxlim:
        thisax.set_xlim(setxlim[0], setxlim[1])
    if setylim:
        thisax.set_ylim(setylim[0], setylim[1])
    return

def autolabelBar(rects, ax, formattext='%.3E'):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                formattext % height,
                ha='center', va='bottom')

class FLIRCamera:
    def __init__(self, saveloc, getBackground=False, backgroundFile='', avgBack = 15, saveopt=True):
        
        # Important parameters
        self.continueRecording = True #must exist
        self.deviceSerialNumber = 0; #placeholder 
        self.params_enum = {
            'AcquisitionMode': 'Continuous',
            'ExposureMode': 'Timed',
            'ExposureAuto': 'Off', #Set to off, Once, continuous
            'GainAuto':'Off', #Set to off, Once, continuous
            'PixelFormat': 'Polarized16'
        }
        self.params_Ifloat = {
            'ExposureTime': 1/10*10**6, #given in us
            #'AcquisitionFrameRate': 5,#htz, Max 10 fps for 16 bit or 24.5 for 8bit
        }
        self.params_bool = {
            'BlackLevelClampingEnable': True, #this performs dark current corrections on the camera
            'GammaEnable': False, #Set false to get linear output
            'ChunkModeActive': True #enables lots of metadata upon image capture
        }

        # Retrieve singleton reference to system object
        self.system = PySpin.System.GetInstance()
        version = self.system.GetLibraryVersion()
        print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))
        
        # Connect to the Camera and initialize
        self.singleCam = self.connectToCamera_Single()
        self.nodemap = self.singleCam.GetNodeMap()
        self.nodemap_tldevice = self.singleCam.GetTLDeviceNodeMap()
        self.sNodemap = self.singleCam.GetTLStreamNodeMap()
        self.configureBufferHandling()
        self.printDeviceInfo()
        self.setCameraParameters()
        self.saveopt = saveopt

        # Test Write Permission for saving images
        if self.saveopt:
            self.saveloc = self.createTimeStampedFolder(saveloc)
            self.testWritePermission()
        
        # Get new dark background if told
        imageshape = [2048,2448]
        self.darkImage = np.zeros(imageshape)
        if getBackground:
            print('Acquiring new dark background')
            self.AcquireImagesContinuous('Capture', num_images=avgBack, DarkBackground=True)
        else:
            if (backgroundFile==''):
                print('Dark Background is zeros array')
                self.badPixelIdx = np.load(
                    'F:/Capasso Group/Depth Sensing Projects/DepthExperiment1/DarkBackgroundFIles/' +
                    '2FPS_DarkBackgroundid-364-time-4021145733680_BadPixelIndex.npy'   
                )
            else:
                print('Using previously loaded dark background')
                self.darkImage = np.load(
                    'F:/Capasso Group/Depth Sensing Projects/DepthExperiment1/DarkBackgroundFIles/' + backgroundFile)
                self.badPixelIdx = self.getBadPixelsFromBackground()

    def connectToCamera_Single(self):
        #Retrieve list of cameras from the system
        cam_list = self.system.GetCameras()
        num_cameras = cam_list.GetSize()
        if num_cameras == 0:
            raise ValueError('No cameras found connected!')
        print('Number of cameras detected: %d' % num_cameras)

        # We are going to make assumption that only a single camera is ever connected...
        singleCam = cam_list.GetByIndex(0)
        singleCam.Init()
        if singleCam.IsInitialized():
            print('Camera Initialized Succesfully')
        else:
            raise ValueError('Camera Did Not Initialize')
            self.terminate()

        return singleCam

    def createTimeStampedFolder(self, saveloc):
        datetime_object = datetime.datetime.now()
        foldername = (saveloc +
            '{}_{}_{}_{}_{}_{}'.format(datetime_object.year, datetime_object.month, datetime_object.day, datetime_object.hour, datetime_object.minute, datetime_object.second) + '/')
        if os.path.exists(foldername):
            shutil.rmtree(foldername)
        os.makedirs(foldername)
        return foldername

    def testWritePermission(self):
        try:
            test_file = open(self.saveloc+'test.txt', 'w+')
        except IOError:
            print('Unable to write to current directory. Please check permissions.')
            self.terminate()
            return

        test_file.close()
        os.remove(test_file.name)
        return

    def printDeviceInfo(self):
        nodeMap = self.nodemap_tldevice
        try: 
            print('*** DEVICE INFORMATION ***\n')
            node_device_information = PySpin.CCategoryPtr(nodeMap.GetNode('DeviceInformation'))

            if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
                features = node_device_information.GetFeatures()
                for feature in features:
                    node_feature = PySpin.CValuePtr(feature)
                    print('%s: %s' % (node_feature.GetName(),
                                    node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))
                    if (node_feature.GetName()=='DeviceSerialNumber'):
                        self.deviceSerialNumber= int(node_feature.ToString())
            else:
                print('Device control information not available.')
        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            self.terminate()
            return False
        print('\n \n ')
        return

    def setCameraParameters(self):
        # do Enumeration List
        for key,val in self.params_enum.items():
            print('Node Key {}; User Target {} '.format(key,val))
            try:
                nodeparam = PySpin.CEnumerationPtr(self.nodemap.GetNode(key))
                # check if mode can be set
                if not PySpin.IsAvailable(nodeparam) or not PySpin.IsWritable(nodeparam):
                    print('Unable to set node mode (enum retrieval). Failed...')

                nodeval = nodeparam.GetEntryByName(val)
                if not PySpin.IsAvailable(nodeval) or not PySpin.IsReadable(nodeval):
                    print('Unable to set node to specified value (entry retrieval). Failed...') 
                nodeval = nodeval.GetValue()
                nodeparam.SetIntValue(nodeval)
            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False

        # do Ifloat
        for key,val in self.params_Ifloat.items():        
            print('Node Key {}; User Target {} '.format(key,val))
            try:
                nodeparam = PySpin.CFloatPtr(self.nodemap.GetNode(key))
                # check if mode can be set
                if not PySpin.IsAvailable(nodeparam) or not PySpin.IsWritable(nodeparam):
                    print('Unable to set node mode (float retrieval). Failed...')

                nodeparam.SetValue(val)
            except PySpin.SpinnakerException as ex:
               print('Error: %s' % ex)
               return False

        #do Boolean
        for key,val in self.params_bool.items():        
            print('Node Key {}; sUser Target {} '.format(key,val))
            try:
                nodeparam = PySpin.CBooleanPtr(self.nodemap.GetNode(key))
                # check if mode can be set
                if not PySpin.IsAvailable(nodeparam) or not PySpin.IsWritable(nodeparam):
                    print('Unable to set node mode (bool retrieval). Failed...')

                nodeparam.SetValue(val)

                if (key == "ChunkModeActive"):
                    if val:
                        print('Image MetaDate True (Enabled Chunks):')
                        self.enableChunkEntries()

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False

        return
            
    def configureBufferHandling(self):
        try:
            # Change bufferhandling mode to NewestOnly
            node_bufferhandling_mode = PySpin.CEnumerationPtr(self.sNodemap.GetNode('StreamBufferHandlingMode'))
            if not PySpin.IsAvailable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
                print('Unable to set stream buffer handling mode.. Aborting...')
                return False

            # Retrieve entry node from enumeration node
            node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
            if not PySpin.IsAvailable(node_newestonly) or not PySpin.IsReadable(node_newestonly):
                print('Unable to set stream buffer handling mode.. Aborting...')
                return False

            # Retrieve integer value from entry node
            node_newestonly_mode = node_newestonly.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_bufferhandling_mode.SetIntValue(node_newestonly_mode)
        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        print('Buffer mode set to newest only')
        return

    def enableChunkEntries(self):
        chunk_selector = PySpin.CEnumerationPtr(self.nodemap.GetNode('ChunkSelector'))
        if not PySpin.IsAvailable(chunk_selector) or not PySpin.IsReadable(chunk_selector):
            print('Unable to retrieve chunk selector. Aborting...\n')
            return False

        entries = [PySpin.CEnumEntryPtr(chunk_selector_entry) for chunk_selector_entry in chunk_selector.GetEntries()]
        print('Enabling chunk entries...')

        # Iterate through our list and select each entry node to enable
        for chunk_selector_entry in entries:
            # Go to next node if problem occurs
            if not PySpin.IsAvailable(chunk_selector_entry) or not PySpin.IsReadable(chunk_selector_entry):
                continue

            chunk_selector.SetIntValue(chunk_selector_entry.GetValue())
            chunk_str = '\t {}:'.format(chunk_selector_entry.GetSymbolic())
            # Retrieve corresponding boolean
            chunk_enable = PySpin.CBooleanPtr(self.nodemap.GetNode('ChunkEnable'))
            # Enable the boolean, thus enabling the corresponding chunk data
            if not PySpin.IsAvailable(chunk_enable):
                print('{} not available'.format(chunk_str))
                result = False
            elif chunk_enable.GetValue() is True:
                print('{} enabled'.format(chunk_str))
            elif PySpin.IsWritable(chunk_enable):
                chunk_enable.SetValue(True)
                print('{} enabled'.format(chunk_str))
            else:
                print('{} not writable'.format(chunk_str))
                result = False

        return

    def getBadPixelsFromBackground(self):
        background=self.darkImage;
        upperboundary = np.mean(background) + 3*np.std(background) 
        badIndex = 1.0-np.array(background > upperboundary).astype(np.int)*1.0

        # lets make sure each polarization image gets the same bad pixels removed?
        # x and y not row and col
        x1,y1 = np.meshgrid(np.arange(0, background.shape[1], 2), np.arange(0, background.shape[0], 2))
        x2,y2 = np.meshgrid(np.arange(1, background.shape[1], 2), np.arange(0, background.shape[0], 2))
        x3,y3 = np.meshgrid(np.arange(0, background.shape[1], 2), np.arange(1, background.shape[0], 2))
        x4,y4 = np.meshgrid(np.arange(1, background.shape[1], 2), np.arange(1, background.shape[0], 2))

        # plot 2d spatial images
        p1 = badIndex[y1,x1] # 90 degrees
        p2 = badIndex[y2,x2] # 45 degrees
        p3 = badIndex[y3,x3] # 135 degrees
        p4 = badIndex[y4,x4] # 0 degrees

        badpix_uniformPol = p1*p2*p3*p4
        badIndex[y1,x1] = badpix_uniformPol
        badIndex[y2,x2] = badpix_uniformPol
        badIndex[y3,x3] = badpix_uniformPol
        badIndex[y4,x4] = badpix_uniformPol

        return badIndex


    def AcquireImagesContinuous(self, mode, num_images=1, DarkBackground=False):
        # Mode must be specified
        # Mode = "Capture" captures and saves num_images sequentially
        # Mode = "Stream" captures and displays in gui the camera feed

        cam = self.singleCam
        i = 0
        print('*** IMAGE ACQUISITION ; MODE {} ***\n'.format(mode))           
        try: 
            cam.BeginAcquisition()
            print('Acquiring images...')
            while i < num_images:
                try:
                    image_result = cam.GetNextImage(1*10**4)
                    # Check completion
                    if image_result.IsIncomplete():
                        print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                    
                    else:
                        image_converted = image_result.Convert(PySpin.PixelFormat_Mono16, PySpin.HQ_LINEAR)
                        image_data = image_converted.GetNDArray()
                        if(not DarkBackground): 
                            # perform background subtraction
                            # unless this acquisition is being called to initialize dark background
                            image_data = (image_data-self.darkImage)*self.badPixelIdx
                            
                        # choose camera aqcuisition mode (in an ugly way)
                        if (mode == 'Capture'):
                            # First get smart file names
                            if (DarkBackground):
                                filename = self.getSmartFileNames(image_result, 'DarkBackgroundid', i)
                                self.darkImage = self.darkImage + image_data*1/num_images;
                                if (i==num_images-1):
                                    np.save(filename, self.darkImage)
                                    # create a file with bad pixels 
                                    badPixelIdx = self.getBadPixelsFromBackground()
                                    np.save(filename+'_BadPixelIndex', badPixelIdx)
                            else: 
                                filename = self.getSmartFileNames(image_result, 'Frameid', i)
                                [p1,p2,p3,p4,deg45Net,deg90Net,deg0Net] = self.getPolarizationSubImages(image_data)
                                datadict = {'rawdata': image_data,
                                        'raw90':p1,
                                        'raw45':p2, 
                                        'raw135':p3, 
                                        'raw0':p4,
                                        'Net45':deg45Net,
                                        'Net90':deg90Net,
                                        'Net0':deg0Net
                                        }
                                with open(filename+'.pkl', 'wb') as handle:
                                    pickle.dump(datadict,handle, protocol=pickle.HIGHEST_PROTOCOL)
                                sci.savemat(filename+'.mat', datadict)
                                
                                if i ==0:   
                                    fig = plt.figure(figsize=(7,10))
                                plt.clf()
                                axisList = addAxis(fig,2,2)
                                self.makePolarizationFourPlot(image_data, fig, axisList)
                                print('Image saved at %s' % filename)
                                plt.savefig(filename+'.png')
                            i += 1

                        elif (mode=='Stream'):
                            if (i == 0):
                                fig = plt.figure(1)     
                            
                            plt.clf()
                            axisList = addAxis(fig,2,2)
                            self.makePolarizationFourPlot(image_data, fig, axisList)

                            if keyboard.is_pressed('ENTER'):
                                print('Program is closing...')
                                # Close figure
                                plt.close('all')             
                                self.continueRecording=False  
                                i = num_images+2
                                             
                        elif (mode=='StreamAnalysis'):
                            if (i == 0):
                                print('Press enter to close the program..')
                                fig = plt.figure(figsize=(10,10))
                                # Close the GUI when close event happens
                                fig.canvas.mpl_connect('close_event', self.handle_close)
                                i = 1
                                start = time.time()
                                timevec = np.array(0);
                                plist = np.array([[0], [0]])
                                streamCutoff = 3; #truncate time 
                           
                            plt.clf()
                            axisList = addAxis(fig,3,3)
                            [timevec, plist] = self.makePolarizationAnalysisPlot(
                                image_data, fig, axisList, start, timevec, plist, streamCutoff
                            )

                            # If user presses enter, close the program
                            if keyboard.is_pressed('ENTER'):
                                #plt.savefig(self.saveloc+'lastframe.png')
                                print('Program is closing...')
                                plt.close('all')                                             
                                self.continueRecording=False  
                                i = num_images+2

                        else:
                            raise IOError('Invalid Mode for Continuous Acquisition given')      

                        #  Release image after each
                        image_result.Release()

                except PySpin.SpinnakerException as ex:
                    print('Error: %s' % ex)
                    return False

            # End camera acquisiton mode after finished getting images
            cam.EndAcquisition()            

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        return True
    
    def AcquireOnTrigger(self): 
        i=0
        while i<100:
            cam = self.singleCam
            try: 
                cam.BeginAcquisition()
                print('Acquiring images...')
                try:
                    image_result = cam.GetNextImage(1*10**4)
                    # Check completion
                    if image_result.IsIncomplete():
                        print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                    else:
                        image_converted = image_result.Convert(PySpin.PixelFormat_Mono16, PySpin.HQ_LINEAR)
                        image_data = image_converted.GetNDArray()
                        image_data = (image_data-self.darkImage)*self.badPixelIdx

                        image_result.Release()
                        # End camera acquisiton mode after finished getting images
                        cam.EndAcquisition()            
                        
                        fig = plt.figure()
                        axisList = addAxis(fig,2,2)                        
                        self.makePolarizationPSFPlot(image_data, fig, axisList)
                        plt.show()

                except PySpin.SpinnakerException as ex:
                    print('Error: %s' % ex)
                    return False
            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False

        return

    def makePolarizationPSFPlot(self, image_data, fig, axisList):
        
        [p1,p2,p3,p4,deg45Net,deg90Net,deg0Net] = self.getPolarizationSubImages(image_data)
        [ymax, xmax] = np.unravel_index(p3.argmax(), p3.shape)
        datadict = {'rawdata': image_data,
                                        'raw90':p1,
                                        'raw45':p2, 
                                        'raw135':p3, 
                                        'raw0':p4,
                                        'Net45':deg45Net,
                                        'Net90':deg90Net,
                                        'Net0':deg0Net
                                        }
        filename = self.saveloc + 'image'
        sci.savemat(filename+'.mat', datadict)
        
        centpixelx = xmax; 
        centpixely = ymax;
        spread = 150;

        im2 = axisList[0].imshow(p4)
        formatPlots(fig, axisList[0], im2, xlabel='', ylabel='',
        title='0 Deg Raw', addcolorbar=True,
        setxlim=[-1*spread+centpixelx, spread+centpixelx], setylim=[-1*spread+centpixely, spread+centpixely])

        im3 = axisList[1].imshow(p1)
        formatPlots(fig, axisList[1], im3, xlabel='', ylabel='',
        title='90 Deg Raw', addcolorbar=True,
        setxlim=[-1*spread+centpixelx, spread+centpixelx], setylim=[-1*spread+centpixely, spread+centpixely])
        
        im4 = axisList[2].imshow(image_data)
        formatPlots(fig, axisList[2], im4, xlabel='', ylabel='',
        title='Raw Image', addcolorbar=True,
        setxlim=[-1*spread+centpixelx, spread+centpixelx], setylim=[-1*spread+centpixely, spread+centpixely])


        return

    def makePolarizationFourPlot(self, image_data, fig, axisList):
        
        [p1,p2,p3,p4,deg45Net,deg90Net,deg0Net] = self.getPolarizationSubImages(image_data)
        [ymax, xmax] = np.unravel_index(p3.argmax(), p3.shape)
        
        centpixelx = xmax; 
        centpixely = ymax;
        spread = 150;

        im1 = axisList[0].imshow(p4, cmap=plt.gray())
        formatPlots(fig, axisList[0], im1, xlabel='', ylabel='',
        title='0 Deg raw', addcolorbar=True,
        setxlim=[-1*spread+centpixelx, spread+centpixelx], setylim=[-1*spread+centpixely, spread+centpixely])

        im2 = axisList[1].imshow(p2)
        formatPlots(fig, axisList[1], im2, xlabel='', ylabel='',
        title='45 Deg raw', addcolorbar=True,
        setxlim=[-1*spread+centpixelx, spread+centpixelx], setylim=[-1*spread+centpixely, spread+centpixely])

        im3 = axisList[2].imshow(p1)
        formatPlots(fig, axisList[2], im3, xlabel='', ylabel='',
        title='90 Deg raw', addcolorbar=True,
        setxlim=[-1*spread+centpixelx, spread+centpixelx], setylim=[-1*spread+centpixely, spread+centpixely])
        
        im4 = axisList[3].imshow(p3)
        formatPlots(fig, axisList[3], im4, xlabel='', ylabel='',
        title='135 Deg Raw', addcolorbar=True,
        setxlim=[-1*spread+centpixelx, spread+centpixelx], setylim=[-1*spread+centpixely, spread+centpixely])

        plt.pause(0.001)
        return

    def makePolarizationAnalysisPlot(
        self, image_data, fig, axisList,start, timevec, plist, streamCutoff):

        [p1,p2,p3,p4,deg45Net,deg90Net,deg0Net] = self.getPolarizationSubImages(image_data)
        [ymax, xmax] = np.unravel_index(p3.argmax(), p3.shape)
        centpixelx = xmax; 
        centpixely = ymax;
        spread = 150;

        # Show raw polarization Pixel Images
        im1 = axisList[0].imshow(p1)
        formatPlots(fig, axisList[0], im1, xlabel='', ylabel='',
        title='90 Deg; tot: %.3E, Max: %.0f'%(np.sum(p1), np.max(p1)), addcolorbar=True,
        setxlim=[-1*spread+centpixelx, spread+centpixelx], setylim=[-1*spread+centpixely, spread+centpixely]
        )


        axisList[1].imshow(p2)
        formatPlots(fig, axisList[1], im1, xlabel='', ylabel='',
        title='45 Deg; total Counts %.3E, Max: %.0f'%(np.sum(p2), np.max(p2)), addcolorbar=True,
        setxlim=[-1*spread+centpixelx, spread+centpixelx], setylim=[-1*spread+centpixely, spread+centpixely]
        )


        axisList[3].imshow(p3)
        formatPlots(fig, axisList[3], im1, xlabel='', ylabel='',
        title='135 Deg; tot_count %.3E, Max: %.0f'%(np.sum(p3), np.max(p3)), addcolorbar=True,
        setxlim=[-1*spread+centpixelx, spread+centpixelx], setylim=[-1*spread+centpixely, spread+centpixely]
        )


        axisList[4].imshow(p4)
        formatPlots(fig, axisList[4], im1, xlabel='', ylabel='',
        title='0 Deg; tot_count %.3E, Max: %.0f'%(np.sum(p4), np.max(p4)), addcolorbar=True,
        setxlim=[-1*spread+centpixelx, spread+centpixelx], setylim=[-1*spread+centpixely, spread+centpixely]
        )

        
        # plot polarization barview
        polarizationvec = ['0 Deg', '45 Deg', '90 Deg', '135 Deg']
        polarizationAdu = [np.sum(p4), np.sum(p2), np.sum(p1), np.sum(p3)]
        rects = axisList[2].bar(polarizationvec, polarizationAdu)
        autolabelBar(rects, axisList[2])
        axisList[2].set_title('Raw Image Sums')


        plist= np.append(plist, np.array([ [np.sum(p4)/np.sum(p1)], [0] ]), axis = 1)
        timevec = np.append(timevec, time.time()-start)
        if len(timevec)>streamCutoff:
            axisList[5].plot(timevec[-streamCutoff:], plist[0,-streamCutoff:], label='P0/P90')
        else:
            axisList[5].plot(timevec[:], plist[0,:], label='P0/P90')
            axisList[5].legend()       
        plt.pause(0.001)
        return timevec, plist

    def getPolarizationSubImages(self, image_data):
        # Define indices to get polarization pixels
        # x and y not row and col
        x1,y1 = np.meshgrid(np.arange(0, image_data.shape[1], 2), np.arange(0, image_data.shape[0], 2))
        x2,y2 = np.meshgrid(np.arange(1, image_data.shape[1], 2), np.arange(0, image_data.shape[0], 2))
        x3,y3 = np.meshgrid(np.arange(0, image_data.shape[1], 2), np.arange(1, image_data.shape[0], 2))
        x4,y4 = np.meshgrid(np.arange(1, image_data.shape[1], 2), np.arange(1, image_data.shape[0], 2))

        p1 = image_data[y1,x1] # 90 degrees
        p2 = image_data[y2,x2] # 45 degrees
        p3 = image_data[y3,x3] # 135 degrees
        p4 = image_data[y4,x4] # 0 degrees

        deg45Net = p2-p3
        deg90Net = p1-0.5*deg45Net
        deg0Net = p4-0.5*deg45Net

        return p1,p2,p3,p4,deg45Net, deg90Net, deg0Net 

    def viewChunkDataofImage(self, spinimage):
        chunk_data = spinimage.GetChunkData()
        # Retrieve exposure time (recorded in microseconds)
        exposure_time = chunk_data.GetExposureTime()
        #print('\tExposure time: {}'.format(exposure_time))

        # Retrieve frame ID
        frame_id = chunk_data.GetFrameID()
        #print('\tFrame ID: {}'.format(frame_id))

        # Retrieve gain; gain recorded in decibels
        gain = chunk_data.GetGain()
        #print('\tGain: {}'.format(gain))

        # Retrieve height; height recorded in pixels
        height = chunk_data.GetHeight()
        #print('\tHeight: {}'.format(height))

        # Retrieve offset X; offset X recorded in pixels
        offset_x = chunk_data.GetOffsetX()
        #print('\tOffset X: {}'.format(offset_x))

        # Retrieve offset Y; offset Y recorded in pixels
        offset_y = chunk_data.GetOffsetY()
        #print('\tOffset Y: {}'.format(offset_y))

        # Retrieve sequencer set active
        sequencer_set_active = chunk_data.GetSequencerSetActive()
        #print('\tSequencer set active: {}'.format(sequencer_set_active))

        # Retrieve timestamp
        timestamp = chunk_data.GetTimestamp()
        #print('\tTimestamp: {}'.format(timestamp))

        # Retrieve width; width recorded in pixels
        width = chunk_data.GetWidth()
        #print('\tWidth: {}'.format(width))

        imagemetadata = {
            'exposure_time': exposure_time,
            'frame_id': frame_id,
            'gain': gain,
            'height': height,
            'width': width,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'sequencer_set_active': sequencer_set_active,
            'timestamp': timestamp
        }

        return imagemetadata

    def getSmartFileNames(self, image_result, tag = 'Frameid', i =0):
        if self.params_bool.get('ChunkModeActive', False):
            metadata = self.viewChunkDataofImage(image_result)
            print('Captured Image metadata: \n ')
            print(metadata)
            filename = self.saveloc+ tag +'-%d-time-%d'%(metadata.get('frame_id'), metadata.get('timestamp'))
        else: 
            filename = self.saveloc+tag + '-%s-%d'%(str(self.deviceSerialNumber), i)
        return filename

    def terminate(self):
        self.singleCam.DeInit()
        del self.singleCam
        self.system.ReleaseInstance()
    
        return

    def handle_close(self):
        # used to terminate continuous stream
        continueRecording = self.continueRecording
        continueRecording = False
        return

    def convertFileImagestoVideo(self, imagetag):
        imagefolder = self.saveloc + imagetag
        img_array = []
        listfiles = glob.glob(imagefolder)
        listfiles.sort(key=os.path.getmtime)

        for filename in listfiles:
            img = cv2.imread(filename)
            height, width, layers = img.shape
            img_array.append(img)
            size = (width,height)

        out = cv2.VideoWriter(
            self.saveloc + 'output.mp4',
            #cv2.VideoWriter_fourcc(*'avc1'),
            cv2.VideoWriter_fourcc(*'mp4v'), 
            2,
            size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

        return  


#saveloc = 'C:/Users/DeanH/Downloads/Background/'
saveloc = 'C:/Users/DeanH/Downloads/PSFExperiment_11_19_2020/'
useDarkBackground = '10FPS_DarkBackgroundid-5577-time-6887460382608.npy'

# initialize camera
FLIRCam = FLIRCamera(
    saveloc, getBackground=False, backgroundFile=useDarkBackground, saveopt=True)
# Run capture mode
FLIRCam.AcquireImagesContinuous(mode='StreamAnalysis', num_images=10)
#FLIRCam.AcquireImagesContinuous(mode='StreamAnalysis', num_images=10)
#FLIRCam.AcquireOnTrigger()
FLIRCam.terminate()

