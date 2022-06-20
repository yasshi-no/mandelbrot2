#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>
#include <mpirxx.h>

namespace mandelbrot_const
{
//�f�B���N�g��
const std::filesystem::path CURRENT_PATH = std::filesystem::current_path();
const std::filesystem::path DATA_PATH = CURRENT_PATH / "my_data";
const std::filesystem::path BIN_DIR = DATA_PATH / "bin";
const std::filesystem::path BIN_CALCED_DIR = BIN_DIR / "calced";
const std::filesystem::path BIN_IMAGE_DIR = BIN_DIR / "image";
const std::filesystem::path IMAGE_DIR = DATA_PATH / "image";
//�摜�̌^
std::string image_type = ".png";

}  // namespace mandelbrot_const

//���f����\���N���X
template <class T>
class Complex
{
public:
    inline Complex(T x, T y) : _x(x), _y(y) {}

    //�����Ƌ��������ꂼ��\������.
    void print()
    {
        std::cout << "_x: " << _x << " _y: " << _y << std::endl;
    }

    //�e���Z�q�̒�`
    inline Complex<T> operator+(const Complex<T>& other) const
    {
        Complex<T> ret(_x + other._x, _y + other._y);
        return ret;
    }

    inline Complex<T> operator-(const Complex<T>& other) const
    {
        Complex<T> ret(_x - other._x, _y - other._y);
        return ret;
    }

    inline Complex<T> operator*(const Complex<T>& other) const
    {
        Complex<T> ret(_x * other._x - _y * other._y, _x * other._y + _y * other._x);
        return ret;
    }

    //�}���f���u���W���ɑ����邩�v�Z��,�����Ȃ��Ɣ��肳�ꂽ�Ƃ��̌v�Z�񐔂�Ԃ�.
    inline int calc_mandelbrot_diverge_time(int time) const
    {
        Complex<T> c(_x, _y);
        Complex<T> z(T(0), T(0));
        for (int i = 0; i < time; i++) {
            z = z * z + c;
            if ((z._x * z._x + z._y * z._y) > 4) {
                return i;
            }
        }
        return -1;
    }

private:
    T _x;
    T _y;
};

//�g�p����f�B���N�g���𐶐�����.
void init_dirs()
{
    std::filesystem::create_directories(mandelbrot_const::DATA_PATH);
    std::filesystem::create_directories(mandelbrot_const::BIN_DIR);
    std::filesystem::create_directories(mandelbrot_const::BIN_CALCED_DIR);
    std::filesystem::create_directories(mandelbrot_const::BIN_IMAGE_DIR);
    std::filesystem::create_directories(mandelbrot_const::IMAGE_DIR);
}

//�ꎞ�t�@�C�����f�B���N�g�����Ə�������.
void delete_temp_dirs()
{
    std::filesystem::remove_all(mandelbrot_const::BIN_DIR);
    std::filesystem::remove_all(mandelbrot_const::IMAGE_DIR);
}

//���b�Z�[�W���e�����s����ׂ����q�˂�.
bool ask_should_execute(std::string msg)
{
    std::string input;
    bool ret;
    std::string yes = "y";
    std::cout << msg << std::endl;
    std::cin >> input;
    ret = (input == yes);
    return ret;
}

//�}���f���u���W���̉摜������킷�N���X
template <class T>
class MandelbrotImage
{
public:
    MandelbrotImage(int width, int height, int calc_time, Complex<T> upper_right, T unit)
        : _width(width), _height(height), _calc_time(calc_time), _upper_right(upper_right), _unit(unit)
    {
    }

    //�}���f���u���W���̉摜�𐶐�����.
    inline cv::Mat create_image(const std::vector<cv::Vec3b>& gradation) const
    {
        cv::Mat image = cv::Mat::zeros(_height, _width, CV_8UC3);

        //gradation�����ƂɊe��f��z�F.
        for (int y = 0; y < _height; y++) {
            cv::Vec3b* pixels = image.ptr<cv::Vec3b>(y);  //y�s�ڂ̐擪��f�̃|�C���^���擾.
            for (int x = 0; x < _width; x++) {
                Complex<T> c = _upper_right + Complex<T>(_unit * x, -_unit * y);
                int diverge_time = c.calc_mandelbrot_diverge_time(_calc_time);
                cv::Vec3b pixel = cv::Vec3b(diverge_time % 255, diverge_time % 255, diverge_time % 255);
                if (diverge_time == -1) {
                } else {
                    pixels[x] = gradation[diverge_time % gradation.size()];
                }
            }
        }
        return image;
    }

private:
    const int _width;               //�摜�̕�
    const int _height;              //�摜�̍���
    const int _calc_time;           //�W���ɑ����邩���肷��܂ł̌v�Z��
    const Complex<T> _upper_right;  //�摜�̉E��̉�f�ɑΉ����镡�f��
    const T _unit;                  //1��f������̈�ӂ̒���
};

//�}���f���u���W���̓��������킷�N���X
template <class T>
class MandelbrotMovie
{
public:
    MandelbrotMovie(const Complex<T> center, const T unit_before, const T unit_after, const T change_ratio, const int width, const int height, const int calc_time)
        : _center(center), _unit_before(unit_before), _unit_after(unit_after), _change_ratio(change_ratio), _width(width), _height(height), _calc_time(calc_time), _image_qty(_calc_image_qty())
    {
    }

    //�}���f���u���W���̉摜�𐶐�����.
    inline void create_images(const std::vector<cv::Vec3b> gradation) const
    {
        int i = 1;
        T unit = _unit_before;

        _skip_calced_images(unit, i);

        while (!_should_end_create_image(unit)) {
            //�e�t���[���̉摜�𐶐�,�ۑ�.
            std::filesystem::path image_path = _calc_image_file_path(i);
            Complex<T> upper_right = _calc_upper_right(unit);
            _create_image(image_path, upper_right, unit, gradation);

            std::cout << "\r"
                      << "�摜�̐���" << i << " / " << _image_qty;

            //���t���[���ւ̏���
            unit *= _change_ratio;
            i++;
        }
    }

    //���񏈗���p���ă}���f���u���W���̉摜�̐���������.
    inline void create_images_multi(const std::vector<cv::Vec3b> gradation, int cpu_num) const
    {
        int i = 1;
        T unit = _unit_before;

        _skip_calced_images(unit, i);

        //����𐶐�����while���[�v
        while (!_should_end_create_image(unit)) {
            std::vector<std::thread> threads;

            //�e�X���b�h�ɏ��������蓖��
            while (!_should_end_create_image(unit) && threads.size() < cpu_num) {
                //�e�t���[���̉摜�𐶐�,�ۑ�.
                std::filesystem::path image_path = _calc_image_file_path(i);
                Complex<T> upper_right = _calc_upper_right(unit);

                threads.push_back(std::thread(&MandelbrotMovie<T>::_create_image, this, image_path, upper_right, unit, gradation));

                //���t���[���ւ̏���
                unit *= _change_ratio;
                i++;
            }

            for (int j = 0; j < threads.size(); j++) {
                threads[j].join();
                std::cout << "\r"
                          << "�摜�̐���" << i - 1 << " / " << _image_qty;
            }
        }
        std::cout << "\r"
                  << "�摜�̐������I�����܂���." << std::endl;
    }

    //�������ꂽ�摜�����Ƃɓ���𐶐�����.
    void create_movie(std::filesystem::path movie_path, int fourcc, double fps)
    {
        int notify_frequency = 10;  //�i���̒ʒm�p�x
        bool isColor = true;        //�F�t�����ۂ�
        cv::VideoWriter writer(movie_path.string(), fourcc, fps, cv::Size(_width, _height), isColor);

        if (!writer.isOpened()) {
            std::cout << "VideoWriter���J�����Ƃ��ł��܂���." << std::endl;
            return;
        }

        //�摜���Ȃ��ē���𐶐�����.
        for (int i = 1; i <= _image_qty; i++) {
            std::filesystem::path image_path = mandelbrot_const::IMAGE_DIR / (std::to_string(i) + ".png");
            cv::Mat image = cv::imread(image_path.string());
            writer << image;
            if (i % notify_frequency == 0) {
                std::cout << "\r"
                          << "����̐���: " << i << " / " << _image_qty;
            }
        }

        std::cout << "\r"
                  << "����̐����͏I�����܂���." << std::endl;

        return;
    }

private:
    //�e�t���[���̉摜�𐶐�,�ۑ�����.
    inline void _create_image(std::filesystem::path image_path, Complex<T> upper_right, T unit, const std::vector<cv::Vec3b> gradation) const
    {
        MandelbrotImage<T> mandelbrot_image(_width, _height, _calc_time, upper_right, unit);
        cv::Mat image = mandelbrot_image.create_image(gradation);
        cv::imwrite(image_path.string(), image);
        return;
    }

    //�摜�t�@�C���̃p�X��Ԃ�.
    inline std::filesystem::path _calc_image_file_path(const int& i) const
    {
        std::filesystem::path image_path = mandelbrot_const::IMAGE_DIR / (std::to_string(i) + mandelbrot_const::image_type);
        return image_path;
    }

    //�E��̉�f�ɑΉ����镡�f�����v�Z����.
    inline Complex<T> _calc_upper_right(const T& unit) const
    {
        Complex<T> upper_right = _center + Complex<T>(-unit * (_width / 2), unit * (_height / 2));
        return upper_right;
    }

    //���������̂ɕK�v�ȉ摜�����v�Z����.
    int _calc_image_qty() const
    {
        T unit = _unit_before;
        int ret = 0;
        while (!_should_end_create_image(unit)) {
            unit *= _change_ratio;
            ret++;
        }

        return ret;
    }

    //�\���ȉ摜�������肷��.
    inline bool _should_end_create_image(const T& unit) const
    {
        bool ret = (unit < _unit_after);
        return ret;
    }

    //�v�Z�ς݂̉摜�̌v�Z���X�L�b�v����.
    void _skip_calced_images(T& unit, int& i) const
    {
        T before_unit = unit;
        //�v�Z�ς݂̉摜���X�L�b�v.
        while (!_should_end_create_image(unit)) {
            if (std::filesystem::exists(_calc_image_file_path(i))) {
                i++;
                before_unit = unit;
                unit *= _change_ratio;
            } else {
                break;
            }
        }

        //�Ō�Ɍv�Z���ꂽ�摜�͔j�����Ă���\�������邽��,�Čv�Z������.
        if (i > 1) {
            i--;
            unit = before_unit;
        }
    }

private:
    const Complex<T> _center;  //����̒��S�̉�f�ɑΉ����镡�f��
    const T _unit_before;      //�ŏ��̂P��f������̕ӂ̒���
    const T _unit_after;       //�Ō�̂P��f������̕ӂ̒���
    const T _change_ratio;     //�P�t���[�����Ƃ�unit�����{�ɂȂ邩
    const int _width;          //����̕�
    const int _height;         //����̍���
    const int _calc_time;      //�W���ɑ����邩���肷��܂ł̌v�Z��
    const int _image_qty;      //���������̂ɕK�v�ȉ摜��
};

cv::Vec3b hsv2bgr(double hue, double saturation, double value)
{
    //hsv��bgr�ɕϊ�����
    //https://yanohirota.com/color-converter/
    double h1 = (int)hue % 60 / 60.0;
    double s1 = saturation / 100;
    double v1 = value / 100;
    uint8_t a = uint8_t(v1 * 255);
    uint8_t b = uint8_t(v1 * (1 - s1) * 255);
    uint8_t c = uint8_t(v1 * (1 - s1 * h1) * 255);
    uint8_t d = uint8_t(v1 * (1 - s1 * (1 - h1)) * 255);
    cv::Vec3b bgr;
    if (saturation == 0) {
        bgr[0] = a;
        bgr[1] = a;
        bgr[2] = a;
    } else if (hue < 60) {
        bgr[0] = a;
        bgr[1] = d;
        bgr[2] = b;
    } else if (hue < 120) {
        bgr[0] = c;
        bgr[1] = a;
        bgr[2] = b;
    } else if (hue < 180) {
        bgr[0] = b;
        bgr[1] = a;
        bgr[2] = d;
    } else if (hue < 240) {
        bgr[0] = b;
        bgr[1] = c;
        bgr[2] = a;
    } else if (hue < 300) {
        bgr[0] = d;
        bgr[1] = b;
        bgr[2] = a;
    } else if (hue >= 300) {
        bgr[0] = a;
        bgr[1] = b;
        bgr[2] = c;
    }
    return bgr;
}

std::vector<cv::Vec3b> calc_gradation(double unit, double saturation, double value)
{
    //�F�̃O���f�[�V����������
    //unit:�F���̕ω�, saturation:�ʓx, value:���x
    std::vector<cv::Vec3b> v;
    for (double hue = 0; hue < 360; hue += unit) {
        v.push_back(hsv2bgr(hue, saturation, value));
    }
    return v;
}

std::vector<cv::Vec3b> add_repaat(std::vector<cv::Vec3b> v)
{
    //�O���f�[�V������܂�Ԃ�
    std::vector<cv::Vec3b> ret;
    int size = (int)v.size();
    for (int i = 0; i < size; i++) {
        ret.push_back(v[i]);
    }
    for (int i = 0; i < size; i++) {
        ret.push_back(v[size - i - 1]);
    }
    return ret;
}

int main()
{
    std::cout << "DATA_PATH: " << mandelbrot_const::DATA_PATH << std::endl;

    //�g�p����f�B���N�g���̏�����
    init_dirs();
    if (ask_should_execute("�v�Z�ς݂̓��e���������܂���?")) {
        delete_temp_dirs();
        init_dirs();
    }

    //������e
    mpf_set_default_prec(1);
    typedef double complex_template_type;                          //complex�̎����Ƌ����ɗp����^
    int width = 1920;                                              //����̕�
    int height = 1080;                                             //����̍���
    int calc_time = 1000;                                          //���U�𒲂ׂ��
    std::vector<cv::Vec3b> gradation = calc_gradation(1, 50, 70);  //�e�_�̔z�F
    gradation = add_repaat(gradation);
    Complex<complex_template_type> center(-1.26222162762384535370226702572022420406, -0.04591700163513884695098681782544085357512);  //����̒��S�̕��f��
    complex_template_type unit_before = 0.001;                                                                                         //����̍ŏ��̏k��
    complex_template_type unit_after = 0.0000000000000001;                                                               //����̍Ō�̏k��
    complex_template_type change_ratio = 0.95;                                                                                       //�P�t���[�����Ƃ̏k�ڂ̕ω�
    std::filesystem::path movie_path = mandelbrot_const::DATA_PATH / "movie2.mp4";                                                   //�������铮��̃p�X
    const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');                                                                  //�R�[�f�b�N
    double fps = 30.0;                                                                                                               //����̃t���[�����[�g

    MandelbrotMovie mandelbrot_movie(center, unit_before, unit_after, change_ratio, width, height, calc_time);

    //�摜�̐���
    //mandelbrot_movie.create_images(gradation);
    mandelbrot_movie.create_images_multi(gradation, 10);

    //����̐���
    mandelbrot_movie.create_movie(movie_path, fourcc, fps);

    return 0;
}