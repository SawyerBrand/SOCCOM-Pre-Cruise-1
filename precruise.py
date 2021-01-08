import numpy as np
import pandas as pd

from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import cm
import cmocean

import xarray as xr

from nbformat import read

from PIL import Image
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.offsetbox import AnchoredText

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import os


c= input("What cruise do you want to see? ")
print(c)

def Meta(cruise):
    if cruise == 'A135_2020':
        LonMin = -30; LonMax = 30; LatMin = -60; LatMax = -20; 
        floats = 'A13/2019_A13.5_Proposed_Floats';stations = 'False';
        csig = 'A135_2020'
    elif cruise == 'Test':
        LonMin = -30; LonMax = 30; LatMin = -60; LatMax = -20; 
        floats = 'test/Test.txt';stations = 'False';
        csig = 'test'
    return(LonMin,LonMax,LatMin,LatMax,floats,stations,csig)

[LonMin,LonMax,LatMin,LatMax,floats,stations,csig] = Meta(c)

def bathy(csig,lonmax,lonmin,latmax,latmin,floats,stations):
    Floats = np.loadtxt(floats) #here, I have created a new variable that loads the float txt file when used

    Lonny = Floats[:,2] #here, I have created a new variable that is made up of all the rows (:) in the third column (2)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.
    Latty = Floats[:,1] #here, I have created a new variable that is made up of all the rows (:) in the second column (1)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.

        
    etopo = xr.open_mfdataset('ETOPO2v2g_f4.nc') #this line reads in the netCDF file within the same directory that
    #contains the etopo data

    etopo.keys() #this line displays the names of the variables contained within the ETOPO file 
    #so you can edit the below code if you need to 

    x = etopo.x # In the ETOPO file I used to build this code, the Lon variable was stored as 'x'
    #so I renamed etopo.x to be x so I didn't have to type so much
    y = etopo.y # In the ETOPO file I used to build this code, the Lat variable was stored as 'y'
    z = etopo.z # In the ETOPO file I used to build this code, the Bathymetry variable was stored as 'z '
    
    #So this is where the boundaries that you set up above come into play. You want to zoom into the area the cruise
    # is taking place but you don't want to just set the boundaries of the plot because it will still be processing 
    # the entire dataset. So instead, you'll limit the dataset itself to the area you want

    Bathy = z[(y>=LatMin)&(y<=LatMax),(x>=LonMin)&(x<=LonMax)] 
    Lon = x[(x>=LonMin)&(x<=LonMax)]
    Lat = y[(y>=LatMin)&(y<=LatMax)]

    #In the above code, you've selected for the indices (which are the numbers that determine the location of the numbers
    # within a matrix or vector or whatever) that correspond to the longitude and latitude set you want. 
    
    
    #Load the Subantarctic Front file:
    saf = np.loadtxt('saf_orsi.txt')
    #the first column will be Longitude (-180,180) format, the second column will be Latitude
    saf_lon = saf[:,0]
    saf_lat = saf[:,1]

    #load the Pacific front file:
    pf = np.loadtxt('pf_orsi.txt')
    #the first column will be Longitude (-180,180) format, the second column will be Latitude
    pf_lon = pf[:,0]
    pf_lat = pf[:,1]

    #load the Pacific front file:
    stf = np.loadtxt('stf_orsi.txt')
    #the first column will be Longitude (-180,180) format, the second column will be Latitude
    stf_lon = stf[:,0]
    stf_lat = stf[:,1]

    #load the Pacific front file:
    sbdy = np.loadtxt('sbdy_orsi.txt')
    #the first column will be Longitude (-180,180) format, the second column will be Latitude
    sbdy_lon = sbdy[:,0]
    sbdy_lat = sbdy[:,1]

    #load the Pacific front file:
    saccf = np.loadtxt('saccf_orsi.txt')
    #the first column will be Longitude (-180,180) format, the second column will be Latitude
    saccf_lon = saccf[:,0]
    saccf_lat = saccf[:,1]
    
    F = loadmat('FebIce.mat')
    FebIce = F['IceConc']
    FLon = F['Lon']
    FLat = F['Lat']
    nlon = F['nlon']
    nlat = F['nlat']
    
    FebIce[FebIce == 0] = 'NaN'
    
    S = loadmat('SeptIce.mat')
    SIce = S['IceConc']
    SLon = S['Lon']
    SLat = S['Lat']
    ß
    SIce[SIce == 0] = 'NaN'
    
    plt.figure(1,(30,20))
    ax = plt.axes(projection=ccrs.PlateCarree(),label='bathy')

    im = plt.contourf(Lon,Lat,Bathy,1000,cmap=cmocean.cm.topo)
    cbar = plt.colorbar(im)
    plt.clim(-8000,8000) #CAUTION editing this colorbar because it will change where the "land"  color starts
    #plt.contour(Lon,Lat,Bathy,100)
    

    plt.plot(Lonny,Latty,'k-')

    plt.plot(Lonny,Latty,'r.',markersize=30,markeredgecolor='green');
    plt.plot(saf_lon,saf_lat,c='0.45',label='SAF',linewidth=4)
    plt.plot(pf_lon,pf_lat,c='0.45',label='PF',linewidth=4)
    plt.plot(saccf_lon,saccf_lat,c='0.45',label='SACCF',linewidth=4)
    plt.plot(sbdy_lon,sbdy_lat,c='0.45',label='SBDY',linewidth=4)
    plt.plot(stf_lon,stf_lat,c='0.45',label='STF',linewidth=4)
    
    ax.add_feature(cfeature.LAND,facecolor='black')
    
    plt.scatter(FLon,FLat,c=FebIce,vmin=0.1,vmax=100,cmap='cool',marker='^',edgecolors='black')
    
    plt.scatter(FLon,FLat,c=FebIce,vmin=0.1,vmax=50,cmap='coolwarm',marker='s',edgecolors='black')
    
    plt.title('Bathymetry for '+csig,fontsize=30)
    

    plt.xlim([LonMin,LonMax])
    plt.ylim([LatMin,LatMax])
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 20, 'color': 'black'}
    gl.ylabel_style = {'size': 20, 'color': 'black'}

    plt.legend()
    
    plt.savefig(csig+'/'+csig+'_Bathy.png')
    Image.open(csig+'/'+csig+'_Bathy.png').save(csig+'/'+csig+'_Bathy.jpg','JPEG')

bathy(csig,LonMax,LonMin,LatMax,LatMin,floats,stations)

def curl(csig,lonmax,lonmin,latmax,latmin,floats,stations):
    Floats = np.loadtxt(floats) #here, I have created a new variable that loads the float txt file when used

    Lonny = Floats[:,2] #here, I have created a new variable that is made up of all the rows (:) in the third column (2)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.
    Latty = Floats[:,1] #here, I have created a new variable that is made up of all the rows (:) in the second column (1)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.

    curl = loadmat('CurlFixed.mat')
    vec = loadmat('VectorFixed.mat')
    
    Curly=np.asarray(curl['Image'][:,:])
    lon=np.asarray(curl['Lon'][:,0])
    lat=np.asarray(curl['Lat'][:,0])
    Curly.shape

    [xx,yy]=np.meshgrid(lon,lat)

    ang = np.asarray(vec['ang'][:,:])
    speed = np.asarray(vec['speed'][:,:])
    lon2 = np.asarray(vec['lon'][:,0])
    lat2 = np.asarray(vec['lat'][:,0])

    [XX,YY]=np.meshgrid(lon2,lat2)

    #So this is where the boundaries that you set up above come into play. You want to zoom into the area the cruise
    # is taking place but you don't want to just set the boundaries of the plot because it will still be processing 
    # the entire dataset. So instead, you'll limit the dataset itself to the area you want

    Curl = Curly[(xx>=LonMin)&(xx<=LonMax)&(yy>=LatMin)&(yy<=LatMax)]
    Lon = lon[(lon>=LonMin)&(lon<=LonMax)]
    Lat = lat[(lat>=LatMin)&(lat<=LatMax)]

    Curl = Curl.reshape(len(Lat),len(Lon))
    #In the above code, you've selected for the indices (which are the numbers that determine the location of the numbers
    # within a matrix or vector or whatever) that correspond to the longitude and latitude set you want. 
    
    Ang = ang[(XX>=LonMin)&(XX<=LonMax)&(YY>=LatMin)&(YY<=LatMax)]
    Speed = speed[(XX>=LonMin)&(XX<=LonMax)&(YY>=LatMin)&(YY<=LatMax)]
    Lon2 = lon2[(lon2>=LonMin)&(lon2<=LonMax)]
    Lat2 = lat2[(lat2>=LatMin)&(lat2<=LatMax)]

    Ang = Ang.reshape(len(Lat2),len(Lon2))*0.0174533
    Speed = Speed.reshape(len(Lat2),len(Lon2))

    U = np.array(Speed*np.cos(Ang))
    V = np.array(Speed*np.sin(Ang))
    
    
    plt.figure(1,(30,20))
    ax = plt.axes(projection=ccrs.PlateCarree(),label='Curl')
    
    plt.plot(Lonny,Latty,'k-')

    im = plt.contourf(Lon,Lat,Curl,1000)
    plt.contour(Lon,Lat,Curl,70,cmap='gist_gray')
    plt.quiver(Lon2,Lat2,U,V)
    cbar = plt.colorbar(im,fraction=0.05)
    
    plt.plot(Lonny,Latty,'r.',markersize=30,markeredgecolor='green');
    
    ax.add_feature(cfeature.LAND,facecolor='black')
    
    plt.title('NCEP Wind Stress Curl for '+csig,fontsize=30)
    
    plt.xlim([LonMin,LonMax])
    plt.ylim([LatMin,LatMax])
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 20, 'color': 'black'}
    gl.ylabel_style = {'size': 20, 'color': 'black'}
    
    plt.savefig(csig+'/'+csig+'_Curl.png')
    Image.open(csig+'/'+csig+'_Curl.png').save(csig+'/'+csig+'_Curl.jpg','JPEG')

    plt.cla()


curl(csig,LonMax,LonMin,LatMax,LatMin,floats,stations)


def altim(csig,lonmax,lonmin,latmax,latmin,floats,stations):
    Floats = np.loadtxt(floats) #here, I have created a new variable that loads the float txt file when used

    Lonny = Floats[:,2] #here, I have created a new variable that is made up of all the rows (:) in the third column (2)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.
    Latty = Floats[:,1] #here, I have created a new variable that is made up of all the rows (:) in the second column (1)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.
    
    f = loadmat('AvisoFixed.mat')
    
    sla = f['SLA']
    adt = f['ADT']
    lats = f['Lats'][:,0]
    lons = f['Lons'][:,0]

    xx = np.linspace(min(lons),max(lons),len(lons))
    yy = np.linspace(min(lats),max(lats),len(lats))

    [XX,YY] = np.meshgrid(xx,yy)
    
    SLA1 = sla[(XX<=LonMax)&(XX>=LonMin)&(YY<=LatMax)&(YY>=LatMin)]
    Lons = lons[(lons<=LonMax)&(lons>=LonMin)]
    Lats = lats[(lats<=LatMax)&(lats>=LatMin)]

    SLA = np.reshape(SLA1,(len(Lats),len(Lons)))
    
    ADT1 = adt[(XX<=LonMax)&(XX>=LonMin)&(YY<=LatMax)&(YY>=LatMin)]
    Lons = lons[(lons<=LonMax)&(lons>=LonMin)]
    Lats = lats[(lats<=LatMax)&(lats>=LatMin)]

    ADT = np.reshape(ADT1,(len(Lats),len(Lons)))
    
    #Load the Subantarctic Front file:
    saf = np.loadtxt('saf_orsi.txt')
    #the first column will be Longitude (-180,180) format, the second column will be Latitude
    saf_lon = saf[:,0]
    saf_lat = saf[:,1]

    #load the Pacific front file:
    pf = np.loadtxt('pf_orsi.txt')
    #the first column will be Longitude (-180,180) format, the second column will be Latitude
    pf_lon = pf[:,0]
    pf_lat = pf[:,1]

    #load the Pacific front file:
    stf = np.loadtxt('stf_orsi.txt')
    #the first column will be Longitude (-180,180) format, the second column will be Latitude
    stf_lon = stf[:,0]
    stf_lat = stf[:,1]

    #load the Pacific front file:
    sbdy = np.loadtxt('sbdy_orsi.txt')
    #the first column will be Longitude (-180,180) format, the second column will be Latitude
    sbdy_lon = sbdy[:,0]
    sbdy_lat = sbdy[:,1]

    #load the Pacific front file:
    saccf = np.loadtxt('saccf_orsi.txt')
    #the first column will be Longitude (-180,180) format, the second column will be Latitude
    saccf_lon = saccf[:,0]
    saccf_lat = saccf[:,1]
    
    plt.figure(1,(30,20))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND,facecolor='black')

    im = plt.contourf(Lons,Lats,SLA,1000,vmin=-0.6,vmax=0.6,cmap='coolwarm',transform=ccrs.PlateCarree())
    plt.contour(Lons,Lats,SLA,70,cmap = 'gray')
    
    cbar2 = plt.colorbar(im,fraction=0.07)
    cbar2.ax.set_title('[m]',fontsize=20)
    cbar2.ax.tick_params(labelsize=20) 
    
    plt.plot(Lonny,Latty,'k-');
    
    plt.plot(Lonny,Latty,'r.',markersize=30,markeredgecolor='green')
    plt.plot(saf_lon,saf_lat,c='0.45',label='SAF',linewidth=3)
    plt.plot(pf_lon,pf_lat,c='0.45',label='PF',linewidth=3)
    plt.plot(saccf_lon,saccf_lat,c='0.45',label='SACCF',linewidth=3)
    plt.plot(sbdy_lon,sbdy_lat,c='0.45',label='SBDY',linewidth=3)
    plt.plot(stf_lon,stf_lat,c='0.45',label='STF',linewidth=3);
    
    plt.title('AVISO Sea Level Anomaly for '+csig,fontsize=30)
    ax.coastlines()
    
    ax.add_feature(cfeature.LAND,facecolor='black')
    plt.xlim([LonMin,LonMax])
    plt.ylim([LatMin,LatMax])
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 20, 'color': 'black'}
    gl.ylabel_style = {'size': 20, 'color': 'black'}


    
    plt.savefig(csig+'/'+csig+'_sla.png')
    Image.open(csig+'/'+csig+'_sla.png').save(csig+'/'+csig+'_sla.jpg','JPEG')
    
    plt.cla()
    plt.figure(2,(30,20))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND,facecolor='black')

    im1 = plt.contourf(Lons,Lats,ADT,1000,vmin=-1.6,vmax=1.6,cmap='coolwarm',transform=ccrs.PlateCarree())
    plt.contour(Lons,Lats,ADT,70,cmap = 'gray')
    
    cbar1 = plt.colorbar(im1,fraction=0.09)
    cbar1.ax.set_title('[m]',fontsize=20)
    cbar1.ax.tick_params(labelsize=20) 
    
    plt.plot(Lonny,Latty,'k-')
    plt.plot(Lonny,Latty,'r.',markersize=30,markeredgecolor='green');
    
    plt.plot(saf_lon,saf_lat,c='0.45',label='SAF',linewidth=3)
    plt.plot(pf_lon,pf_lat,c='0.45',label='PF',linewidth=3)
    plt.plot(saccf_lon,saccf_lat,c='0.45',label='SACCF',linewidth=3)
    plt.plot(sbdy_lon,sbdy_lat,c='0.45',label='SBDY',linewidth=3)
    plt.plot(stf_lon,stf_lat,c='0.45',label='STF',linewidth=3);
    
    plt.title('AVISO Absolute Dynamic Topography for '+csig,fontsize=30)
    ax.coastlines()
    
    ax.add_feature(cfeature.LAND,facecolor='black')
    plt.xlim([LonMin,LonMax])
    plt.ylim([LatMin,LatMax])
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 20, 'color': 'black'}
    gl.ylabel_style = {'size': 20, 'color': 'black'}

    
    plt.savefig(csig+'/'+csig+'_adt.png')
    Image.open(csig+'/'+csig+'_adt.png').save(csig+'/'+csig+'_adt.jpg','JPEG')

    plt.cla()

altim(csig,LonMax,LonMin,LatMax,LatMin,floats,stations)

plt.clf()
plt.cla()

def sst(csig,lonmax,lonmin,latmax,latmin,floats,stations):
    Floats = np.loadtxt(floats) #here, I have created a new variable that loads the float txt file when used

    Lonny = Floats[:,2] #here, I have created a new variable that is made up of all the rows (:) in the third column (2)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.
    Latty = Floats[:,1] #here, I have created a new variable that is made up of all the rows (:) in the second column (1)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.
    
    sst = loadmat('SST.mat')
    sst.keys()
    
    lats = sst['lat'][:,0];
    lons = sst['lon'][:,0];
    ssts = sst['sst'][:,:,0];
    SSTs = ssts.transpose()

    xx = np.linspace(min(lons),max(lons),len(lons))
    yy = np.linspace(min(lats),max(lats),len(lats))

    [XX,YY] = np.meshgrid(xx,yy)
    
    SST = SSTs[(XX<=LonMax)&(XX>=LonMin)&(YY<=LatMax)&(YY>=LatMin)]
    Lons = lons[(lons<=LonMax)&(lons>=LonMin)]
    Lats = lats[(lats<=LatMax)&(lats>=LatMin)]
    SSTS = np.reshape(SST,(len(Lats),len(Lons)))
    
    
    plt.figure(1,(30,20))
    ax = plt.axes(projection=ccrs.PlateCarree(),label='sst')
    
    im2 = plt.contourf(Lons,Lats,SSTS,1000,cmap='coolwarm')
    plt.contour(Lons,Lats,SSTS,70,cmap='gray')
    cbar2 = plt.colorbar(im2, fraction=0.05)
    plt.plot(Lonny,Latty,'k-')
    
    plt.plot(Lonny,Latty,'r.',markersize=30,markeredgecolor='green');
    
    plt.title('NOAA OI Sea Surface Temperature for '+csig,fontsize=30)
    
    plt.xlim([LonMin,LonMax])
    plt.ylim([LatMin,LatMax])
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    
    
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 20, 'color': 'black'}
    gl.ylabel_style = {'size': 20, 'color': 'black'}

    plt.legend()
    
    plt.savefig(csig+'/'+csig+'_SST.png')
    Image.open(csig+'/'+csig+'_SST.png').save(csig+'/'+csig+'_SST.jpg','JPEG')

sst(csig,LonMax,LonMin,LatMax,LatMin,floats,stations)

def MLDS(csig,lonmax,lonmin,latmax,latmin,floats,stations):
    Floats = np.loadtxt(floats) #here, I have created a new variable that loads the float txt file when used

    Lonny = Floats[:,2] #here, I have created a new variable that is made up of all the rows (:) in the third column (2)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.
    Latty = Floats[:,1] #here, I have created a new variable that is made up of all the rows (:) in the second column (1)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.
    
    f = xr.open_mfdataset('Argo_mixedlayers_monthlyclim_12112019.nc')
    
    lat = f['lat'].values
    lon = f['lon'].values
    mld = f['mld_da_mean'][:,:,8].values
    
    [xx,yy]=np.meshgrid(lon,lat)

    #So this is where the boundaries that you set up above come into play. You want to zoom into the area the cruise
    # is taking place but you don't want to just set the boundaries of the plot because it will still be processing 
    # the entire dataset. So instead, you'll limit the dataset itself to the area you want

    MLD = mld[(xx>=LonMin)&(xx<=LonMax)&(yy>=LatMin)&(yy<=LatMax)]
    Lon = lon[(lon>=LonMin)&(lon<=LonMax)]
    Lat = lat[(lat>=LatMin)&(lat<=LatMax)]
    
    MLD1 = MLD.reshape(len(Lat),len(Lon))
    
    plt.figure(1,(30,20))
    ax = plt.axes(projection=ccrs.PlateCarree(),label='MLD')
    im = ax.pcolor(Lon,Lat,MLD1)
    
    ax.plot(Lonny,Latty,'k-')

    ax.plot(Lonny,Latty,'r.',markersize=50,markeredgecolor='green');

    ax.coastlines()

    g = ax.gridlines(draw_labels=True,linestyle='--') #add gridlines to the map
    
    cbar = plt.colorbar(im,fraction=0.03)
    cbar.ax.tick_params(labelsize=24) 

    g.ylabels_right = False
    g.xlabels_top = False
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    g.xformatter = LONGITUDE_FORMATTER
    g.yformatter = LATITUDE_FORMATTER
    g.xlabel_style = {'size': 24}
    g.ylabel_style = {'size': 24}
    
    plt.title('September Mixed Layer Depth (Argo Climatology, Holte et al) for '+csig,fontsize=30)
    plt.savefig(csig+'/'+csig+'_MLD.png')
    #Image.open(csig+'/'+csig+'_MLD.png').save(csig+'/'+csig+'_MLD.jpg','JPEG');

MLDS(csig,LonMax,LonMin,LatMax,LatMin,floats,stations)

def CO2_plot(csig,lonmax,lonmin,latmax,latmin,floats,stations):
    Floats = np.loadtxt(floats) #here, I have created a new variable that loads the float txt file when used

    Lonny = Floats[:,2] #here, I have created a new variable that is made up of all the rows (:) in the third column (2)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.
    Latty = Floats[:,1] #here, I have created a new variable that is made up of all the rows (:) in the second column (1)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.
    
    c = loadmat('spco2_mean_2005-2015_MPI_SOM-FFN_v2016.mat')
    
    co2 = c['fgco2_smoothed_Mean_2005to2015']
    lons = c['lon'][:,0]
    lats = c['lat'][:,0]
    co2 = co2.transpose()

    xx = np.linspace(min(lons),max(lons),len(lons))
    yy = np.linspace(min(lats),max(lats),len(lats))

    [XX,YY] = np.meshgrid(xx,yy)
    
    co2s = co2[(XX<=LonMax)&(XX>=LonMin)&(YY<=LatMax)&(YY>=LatMin)]
    Lons = lons[(lons<=LonMax)&(lons>=LonMin)]
    Lats = lats[(lats<=LatMax)&(lats>=LatMin)]

    CO2 = np.reshape(co2s,(len(Lats),len(Lons)))
    
    plt.figure(1,(30,20))
    ax = plt.axes(projection=ccrs.PlateCarree(),label='CO2')

    im = plt.contourf(Lons,Lats,CO2,1000,vmin=-3,vmax=3,cmap='bwr')
    plt.contour(Lons,Lats,CO2,70,cmap = 'gray')
    
    cbar = plt.colorbar(im,fraction=0.05)
    cbar.ax.tick_params(labelsize=20) 
    
    plt.plot(Lonny,Latty,'k-')
    plt.plot(Lonny,Latty,'r.',markersize=30,markeredgecolor='green');
    
    ax.add_feature(cfeature.LAND,facecolor='black')
    
    plt.title('Air-sea CO2 Flux (Landschutzer annual mean) for '+csig,fontsize=30)
    
    plt.xlim([LonMin,LonMax])
    plt.ylim([LatMin,LatMax])
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 20, 'color': 'black'}
    gl.ylabel_style = {'size': 20, 'color': 'black'}

    plt.legend()
    
    plt.savefig(csig+'/'+csig+'_CO2.png')
    Image.open(csig+'/'+csig+'_CO2.png').save(csig+'/'+csig+'_CO2.jpg','JPEG')

CO2_plot(csig,LonMax,LonMin,LatMax,LatMin,floats,stations)

def fresh_plot(csig,lonmax,lonmin,latmax,latmin,floats,stations):
    Floats = np.loadtxt(floats) #here, I have created a new variable that loads the float txt file when used

    Lonny = Floats[:,2] #here, I have created a new variable that is made up of all the rows (:) in the third column (2)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.
    Latty = Floats[:,1] #here, I have created a new variable that is made up of all the rows (:) in the second column (1)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.
    
    fresh = loadmat('FreshFixed.mat')
    
    FRE = fresh['Fresh']
    lats = fresh['Lat'][:,0]
    lons = fresh['Lon'][:,0]


    xx = np.linspace(min(lons),max(lons),len(lons))
    yy = np.linspace(min(lats),max(lats),len(lats))

    [XX,YY] = np.meshgrid(xx,yy)
    
    xx = np.linspace(min(lons),max(lons),len(lons))
    yy = np.linspace(min(lats),max(lats),len(lats))

    [XX,YY] = np.meshgrid(xx,yy)
    
    fresh = FRE[(XX<=LonMax)&(XX>=LonMin)&(YY<=LatMax)&(YY>=LatMin)]
    Lons = lons[(lons<=LonMax)&(lons>=LonMin)]
    Lats = lats[(lats<=LatMax)&(lats>=LatMin)]

    Fresh = np.reshape(fresh,(len(Lats),len(Lons)))

    plt.figure(1,(30,20))
    ax = plt.axes(projection=ccrs.PlateCarree(),label='fresh')
    
    ax.add_feature(cfeature.LAND,facecolor='black')

    im = plt.contourf(Lons,Lats,Fresh,600,vmin=-35,vmax=35,cmap='bwr')
    plt.contour(Lons,Lats,Fresh,70,cmap = 'gray')
    
    cbar = plt.colorbar(im,fraction=0.05)
    cbar.ax.set_title('[W/m2]',fontsize=20)
    cbar.ax.tick_params(labelsize=20) 
    
    plt.plot(Lonny,Latty,'k-')
    plt.plot(Lonny,Latty,'r.',markersize=30,markeredgecolor='green');
    
    
    plt.title('LY09 Annual Mean Fresh Water Flux [Equivalent W/m2] for '+csig,fontsize=30)
    
    plt.xlim([LonMin,LonMax])
    plt.ylim([LatMin,LatMax])
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 20, 'color': 'black'}
    gl.ylabel_style = {'size': 20, 'color': 'black'}

    
    plt.savefig(csig+'/'+csig+'_Fresh.png')
    Image.open(csig+'/'+csig+'_Fresh.png').save(csig+'/'+csig+'_Fresh.jpg','JPEG')

fresh_plot(csig,LonMax,LonMin,LatMax,LatMin,floats,stations)

def buoy_plot(csig,lonmax,lonmin,latmax,latmin,floats,stations):
    Floats = np.loadtxt(floats) #here, I have created a new variable that loads the float txt file when used

    Lonny = Floats[:,2] #here, I have created a new variable that is made up of all the rows (:) in the third column (2)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.
    Latty = Floats[:,1] #here, I have created a new variable that is made up of all the rows (:) in the second column (1)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.
    
    buoy = loadmat('BuoyancyFixed.mat')
    
    B = buoy['buoy']
    lats = buoy['lat'][:,0]
    lons = buoy['lon'][:,0]


    xx = np.linspace(min(lons),max(lons),len(lons))
    yy = np.linspace(min(lats),max(lats),len(lats))

    [XX,YY] = np.meshgrid(xx,yy)
    
    xx = np.linspace(min(lons),max(lons),len(lons))
    yy = np.linspace(min(lats),max(lats),len(lats))

    [XX,YY] = np.meshgrid(xx,yy)
    
    buoyy = B[(XX<=LonMax)&(XX>=LonMin)&(YY<=LatMax)&(YY>=LatMin)]
    Lons = lons[(lons<=LonMax)&(lons>=LonMin)]
    Lats = lats[(lats<=LatMax)&(lats>=LatMin)]

    Buoy = np.reshape(buoyy,(len(Lats),len(Lons)))

    plt.show()
    plt.figure(1,(30,20))
    ax = plt.axes(projection=ccrs.PlateCarree(),label='buoy')

    im = plt.contourf(Lons,Lats,Buoy,1000,vmin=-160,vmax=160,cmap='seismic_r')
    plt.contour(Lons,Lats,Buoy,70,cmap='gist_gray')
    plt.contour(Lons,Lats,Buoy,cmap = 'gray')
    
    cbar = plt.colorbar(im,fraction=0.05)
    cbar.ax.set_title('[W/m2]',fontsize=20)
    cbar.ax.tick_params(labelsize=20) 
    
    plt.plot(Lonny,Latty,'k-')
    plt.plot(Lonny,Latty,'r.',markersize=30,markeredgecolor='green');
    
    ax.add_feature(cfeature.LAND,facecolor='black')
    
    plt.title('LY09 Annual Mean Buoyancy Flux [Equivalent W/m^2] for '+csig,fontsize=30)
    
    plt.xlim([LonMin,LonMax])
    plt.ylim([LatMin,LatMax])
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 20, 'color': 'black'}
    gl.ylabel_style = {'size': 20, 'color': 'black'}

    
    plt.savefig(csig+'/'+csig+'_Buoy.png')
    Image.open(csig+'/'+csig+'_Buoy.png').save(csig+'/'+csig+'_Buoy.jpg','JPEG')

buoy_plot(csig,LonMax,LonMin,LatMax,LatMin,floats,stations)

def heat_plot(csig,lonmax,lonmin,latmax,latmin,floats,stations):
    Floats = np.loadtxt(floats) #here, I have created a new variable that loads the float txt file when used

    Lonny = Floats[:,2] #here, I have created a new variable that is made up of all the rows (:) in the third column (2)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.
    Latty = Floats[:,1] #here, I have created a new variable that is made up of all the rows (:) in the second column (1)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.
    
    heat = loadmat('HeatFixed.mat')
    
    H = heat['heat']
    lats = heat['lat'][:,0]
    lons = heat['lon'][:,0]

    xx = np.linspace(min(lons),max(lons),len(lons))
    yy = np.linspace(min(lats),max(lats),len(lats))

    [XX,YY] = np.meshgrid(xx,yy)
    
    hh = H[(XX<=LonMax)&(XX>=LonMin)&(YY<=LatMax)&(YY>=LatMin)]
    Lons = lons[(lons<=LonMax)&(lons>=LonMin)]
    Lats = lats[(lats<=LatMax)&(lats>=LatMin)]

    Heat = np.reshape(hh,(len(Lats),len(Lons)))

    plt.figure(1,(30,20))
    ax = plt.subplot(projection=ccrs.PlateCarree(),label='heat')

    im = plt.contourf(Lons,Lats,Heat,1000,vmin=-120,vmax=120,cmap='seismic')
    plt.contour(Lons,Lats,Heat,70,cmap = 'gray')
    
    cbar = plt.colorbar(im,fraction=0.05)
    cbar.ax.set_title('[W/m2]',fontsize=20)
    cbar.ax.tick_params(labelsize=20) 
    
    plt.plot(Lonny,Latty,'k-')
    plt.plot(Lonny,Latty,'r.',markersize=30,markeredgecolor='green');
    
    plt.title('LY09 Annual Mean Heat Flux [W/m2] for '+csig,fontsize=30)
    ax.coastlines(linewidth=5)    
    ax.add_feature(cfeature.LAND,facecolor='black')
    
    plt.xlim([LonMin,LonMax])
    plt.ylim([LatMin,LatMax])
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 20, 'color': 'black'}
    gl.ylabel_style = {'size': 20, 'color': 'black'}

    
    plt.savefig(csig+'/'+csig+'_Heat.png')
    Image.open(csig+'/'+csig+'_Heat.png').save(csig+'/'+csig+'_Heat.jpg','JPEG')

heat_plot(csig,LonMax,LonMin,LatMax,LatMin,floats,stations)

def chloro(csig,lonmax,lonmin,latmax,latmin,floats,stations):
    Floats = np.loadtxt(floats) #here, I have created a new variable that loads the float txt file when used

    Lonny = Floats[:,2] #here, I have created a new variable that is made up of all the rows (:) in the third column (2)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.
    Latty = Floats[:,1] #here, I have created a new variable that is made up of all the rows (:) in the second column (1)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.

    Floats = np.loadtxt(floats) #here, I have created a new variable that loads the float txt file when used

    Lonny = Floats[:,2] #here, I have created a new variable that is made up of all the rows (:) in the third column (2)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.
    Latty = Floats[:,1] #here, I have created a new variable that is made up of all the rows (:) in the second column (1)
    # of the float text file. The [:,2] means the indices that correspond will all the rows of the third column.
    
    chl = loadmat('Chloro.mat')
    chl.keys()

    lats = chl['lat'][:,0];
    lons = chl['lon'][:,0];
    chls = chl['chlor']
    Chl1 = chls.transpose();
    Chl2 = chls.transpose();
    
    xx = np.linspace(min(lons),max(lons),len(lons))
    yy = np.linspace(min(lats),max(lats),len(lats))

    yy = -1*yy;

    [XX,YY] = np.meshgrid(xx,yy)

    Chl1[Chl1>1] = np.nan 
    Chl2[Chl2>2] = np.nan 

    Chls1 = Chl1[(XX<=LonMax)&(XX>=LonMin)&(YY<=LatMax)&(YY>=LatMin)]
    Lons = lons[(lons<=LonMax)&(lons>=LonMin)]
    Lats = lats[(lats<=LatMax)&(lats>=LatMin)]
    ChlS1 = np.reshape(Chls1,(len(Lats),len(Lons)))

    Chls2 = Chl2[(XX<=LonMax)&(XX>=LonMin)&(YY<=LatMax)&(YY>=LatMin)]
    ChlS2 = np.reshape(Chls2,(len(Lats),len(Lons)))

    plt.figure(1,(30,20))
    ax = plt.axes(projection=ccrs.PlateCarree(),label='chloro')

    im = plt.contourf(Lons,Lats,ChlS1,1000)
    cbar = plt.colorbar(im,fraction=0.05)

    plt.plot(Lonny,Latty,'k-')
    plt.plot(Lonny,Latty,'r.',markersize=30,markeredgecolor='green');
    
    ax.add_feature(cfeature.LAND,facecolor='black')
    
    plt.title('MODIS Chlorophyll for '+csig,fontsize=30)

    plt.xlim([LonMin,LonMax])
    plt.ylim([LatMin,LatMax])
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 20, 'color': 'black'}
    gl.ylabel_style = {'size': 20, 'color': 'black'}

    
    plt.savefig(csig+'/'+csig+'_Chloro.png')
    Image.open(csig+'/'+csig+'_Chloro.png').save(csig+'/'+csig+'_Chloro.jpg','JPEG')

chloro(csig,LonMax,LonMin,LatMax,LatMin,floats,stations)