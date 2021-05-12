#include <iostream>
#include <vector>
using namespace std;

#include <rtac_base/files.h>
using namespace rtac;

std::vector<uint8_t> image_data(int width, int height)
{
    std::vector<uint8_t> res(3*width*height);
    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
            unsigned int bw = (w + h) & 0x1;

            if(bw) {
                res[3*(width*h + w)]     = 0;
                res[3*(width*h + w) + 1] = 255;
                res[3*(width*h + w) + 2] = 255;
            }
            else {
                res[3*(width*h + w)]     = 255;
                res[3*(width*h + w) + 1] = 124;
                res[3*(width*h + w) + 2] = 0;
            }
        }
    }
    return res;
}

int main()
{
    unsigned int W = 8, H = 8;
    auto outputData = image_data(W,H);
    
    files::write_ppm("out.ppm", W, H, (const char*)outputData.data());

    std::vector<uint8_t> inputData;
    size_t Win = 0, Hin = 0;
    
    // std::string path = "out.ppm";
    std::string path = files::find_one(".*mummy-orthoimage-halfResolution.ppm");
    cout << "path : " << path << endl;
    files::read_ppm(path, Win, Hin, inputData);

    files::write_ppm("check.ppm", Win, Hin, (const char*)inputData.data());

    return 0;
}
