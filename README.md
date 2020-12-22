# SOCCOM-Pre-Cruise

This is the repository for SOCCOM's pre-cruise code. The purpose of the code is to automate the pre-cruise reports for SOCCOM's Observations department. 

### Pre-Cruise-Creation.ipynb
This is the main jupyter notebook. Within it, all the other notebooks used are called and run, so the user only has to run one jupyter notebook for each pre-cruise. The only part of this notebook that needs editing to run is the fourth code cell, containing "os.mkdir(csis)". When you first run this notebook for each cruise, uncomment this to create a unique folder under which all the plots will be saved. After that initial run for each cruise, comment it out. 

### MetaInfo.ipynb
This is the jupyter notebook in which all unique info for each cruise will be stored. THIS SHOULD BE THE ONLY ONE YOU NEED TO EDIT FOR A BASIC CRUISE. The information is stored in an if-else statement, so when you add a new cruise, make sure to copy the format. This way, when run, a unique set of parameters will be stored to be used in Pre-Cruise-Creations.ipynb. 

The editable info in this notebook is: the lon/lat boundaries, the "csig" or cruise signifier that will be in the title of each plot and the folder you want to create, the name of the float location text file you want to use (entered as a string containing the folder you will be creating, as you should store your text there; 'csig/csig_Float_Locations.txt' should be the format), and then the same for the station location file. If you are not going to use a float text file or a station text file, replace their path names with 'false'.

### Bathy.ipynb

### Curl.ipynb

### Altim.ipynb

### SST.ipynb

### Chloro.ipynb

### MLD.ipynb

### CO2.ipynb

### Fresh.ipynb

### Buoyancy.ipynb

### Heat.ipynb
