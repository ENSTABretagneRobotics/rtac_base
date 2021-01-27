#include <iostream>
using namespace std;

#include <rtac_base/types/Mesh.h>

int main()
{
    auto cube = rtac::types::Mesh<>::cube();
    cout << "Cube0 :\n" << cube << endl;

    cube.export_ply("cube.ply");

    auto cube1 = rtac::types::Mesh<>::from_ply("cube.ply");
    cout << "Cube1 :\n" << cube1 << endl;

    return 0;
}

