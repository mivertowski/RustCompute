Showcase app

RingKernel backend, Iced as GUI framework

Application with GUI that:

- shows a 2D matrix (e.g., 8x16)
- each cell is an actor and represents a quantized 2d space vortex
- each cell is connected to all the neighbor cells
- each cell passes energy/pressure changes to it's neighbor
- it shall simulate sound wave propagation in a (quantized) room
- based on real physics and filtering
- the user can "interact" with the 2d room in clicking somewhere what is treated as room impulse (e.g., dirac)
- the gui should show how the waves/impulses are propagated in a visual appealing way
- simulates the k2k message passing and how the gpu actor net is working in a nice way
- there should be a slider to change the speed of sound value to also accept very low arbitrary values (makes it easier to see the animations better)
- we need to test different "room sizes" for performance and accuracy
