Currentdir = pwd;
cd ('C:\Program Files\COMSOL\COMSOL54\Multiphysics\bin\win64');
system('comsolserver.exe &');
cd(Currentdir);

Currentdir = pwd;
cd('C:\Program Files\COMSOL\COMSOL54\Multiphysics\mli');
mphstart(2036);
cd(Currentdir);
