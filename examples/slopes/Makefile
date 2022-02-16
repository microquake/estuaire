NPROCESS = 1
SITE_DIR = ../../site_scons

all : .PHONY

.PHONY :
	scons --site-dir=${SITE_DIR} -j${NPROCESS}

tree :
	scons --site-dir=${SITE_DIR} --tree=all

clean :
	scons --site-dir=../site_scons -c
	#rm -rf ./build -
