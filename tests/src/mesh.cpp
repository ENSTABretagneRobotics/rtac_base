#include <iostream>
using namespace std;

#include <rtac_base/types/Mesh.h>
using namespace rtac::types;

int main()
{
    Mesh<> mesh0;
    cout << mesh0 << endl;

    auto cube = Mesh<>::cube();
    cout << "Cube0 :\n" << *cube << endl;

    cube->export_ply("cube.ply", true);

    auto cube1 = Mesh<>::from_ply("cube.ply");
    cout << "Cube1 :\n" << *cube1 << endl;

    return 0;
}

