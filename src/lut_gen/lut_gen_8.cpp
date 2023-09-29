#include <iostream>
#include <vector>
using namespace std;

struct Node {
    int value;
    int prev;
    int next;
};

int main()
{
    std::vector<Node> nodes(8);
    for(int i=0; i<8; i++){
        nodes[i].value = (1 << i);
        nodes[i].prev = i-1;
        nodes[i].next = i+1;
    }
    nodes[0].prev = 7;
    nodes[7].next = 0;

    uint8_t LUT[8*2*16] = {0};

    for(int i=0; i<8; i++){ // 8 ori
        for(int m=0; m<2; m++){ // 2 seg
            for(int n=0; n<16; n++){ // 16 index

                if(n==0){ // no ori
                   LUT[n+m*16+i*16*2] = 0;
                   continue;
                }

                int res = (n << (m * 4));
                auto current_node_go_forward = nodes[i];
                auto current_node_go_back = nodes[i];
                int angle_diff = 0;
                while(1){
                    if((current_node_go_forward.value & res) > 0 ||
                       (current_node_go_back.value & res) > 0){
                        break;
                    }else{
                        current_node_go_back = nodes[current_node_go_back.prev];
                        current_node_go_forward = nodes[current_node_go_forward.next];
                        angle_diff ++;
                    }
                }
                LUT[n+m*16+i*16*2] = 4 - angle_diff;
            }
        }
    }

    for(int i=0; i<8; i++){
        for(int m=0; m<32; m++){
            cout << int(LUT[i*32 + m]) << ", ";
        }
        cout << "\n";
    }

    return 0;
}

