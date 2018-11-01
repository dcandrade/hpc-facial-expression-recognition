all: .serial .multi-thread

.serial:
	cd serial/
	make setup-experiments
	make experiments
	cd ..
	
.multi-thread:
	cd multi-thread
	make setup-experiments
	make experiments
	cd ..
