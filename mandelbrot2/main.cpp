#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>
#include <mpirxx.h>

namespace mandelbrot_const
{
//ディレクトリ
const std::filesystem::path CURRENT_PATH = std::filesystem::current_path();
const std::filesystem::path DATA_PATH = CURRENT_PATH / "my_data";
const std::filesystem::path BIN_DIR = DATA_PATH / "bin";
const std::filesystem::path BIN_CALCED_DIR = BIN_DIR / "calced";
const std::filesystem::path BIN_IMAGE_DIR = BIN_DIR / "image";
const std::filesystem::path IMAGE_DIR = DATA_PATH / "image";
//画像の型
std::string image_type = ".png";

}  // namespace mandelbrot_const

//複素数を表すクラス
template <class T>
class Complex
{
public:
    inline Complex(T x, T y) : _x(x), _y(y) {}

    //実部と虚部をそれぞれ表示する.
    void print()
    {
        std::cout << "_x: " << _x << " _y: " << _y << std::endl;
    }

    //各演算子の定義
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

    //マンデルブロ集合に属するか計算し,属さないと判定されたときの計算回数を返す.
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

//使用するディレクトリを生成する.
void init_dirs()
{
    std::filesystem::create_directories(mandelbrot_const::DATA_PATH);
    std::filesystem::create_directories(mandelbrot_const::BIN_DIR);
    std::filesystem::create_directories(mandelbrot_const::BIN_CALCED_DIR);
    std::filesystem::create_directories(mandelbrot_const::BIN_IMAGE_DIR);
    std::filesystem::create_directories(mandelbrot_const::IMAGE_DIR);
}

//一時ファイルをディレクトリごと消去する.
void delete_temp_dirs()
{
    std::filesystem::remove_all(mandelbrot_const::BIN_DIR);
    std::filesystem::remove_all(mandelbrot_const::IMAGE_DIR);
}

//メッセージ内容を実行するべきか尋ねる.
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

//マンデルブロ集合の画像をあらわすクラス
template <class T>
class MandelbrotImage
{
public:
    MandelbrotImage(int width, int height, int calc_time, Complex<T> upper_right, T unit)
        : _width(width), _height(height), _calc_time(calc_time), _upper_right(upper_right), _unit(unit)
    {
    }

    //マンデルブロ集合の画像を生成する.
    inline cv::Mat create_image(const std::vector<cv::Vec3b>& gradation) const
    {
        cv::Mat image = cv::Mat::zeros(_height, _width, CV_8UC3);

        //gradationをもとに各画素を配色.
        for (int y = 0; y < _height; y++) {
            cv::Vec3b* pixels = image.ptr<cv::Vec3b>(y);  //y行目の先頭画素のポインタを取得.
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
    const int _width;               //画像の幅
    const int _height;              //画像の高さ
    const int _calc_time;           //集合に属するか判定するまでの計算回数
    const Complex<T> _upper_right;  //画像の右上の画素に対応する複素数
    const T _unit;                  //1画素当たりの一辺の長さ
};

//マンデルブロ集合の動画をあらわすクラス
template <class T>
class MandelbrotMovie
{
public:
    MandelbrotMovie(const Complex<T> center, const T unit_before, const T unit_after, const T change_ratio, const int width, const int height, const int calc_time)
        : _center(center), _unit_before(unit_before), _unit_after(unit_after), _change_ratio(change_ratio), _width(width), _height(height), _calc_time(calc_time), _image_qty(_calc_image_qty())
    {
    }

    //マンデルブロ集合の画像を生成する.
    inline void create_images(const std::vector<cv::Vec3b> gradation) const
    {
        int i = 1;
        T unit = _unit_before;

        _skip_calced_images(unit, i);

        while (!_should_end_create_image(unit)) {
            //各フレームの画像を生成,保存.
            std::filesystem::path image_path = _calc_image_file_path(i);
            Complex<T> upper_right = _calc_upper_right(unit);
            _create_image(image_path, upper_right, unit, gradation);

            std::cout << "\r"
                      << "画像の生成" << i << " / " << _image_qty;

            //次フレームへの処理
            unit *= _change_ratio;
            i++;
        }
    }

    //並列処理を用いてマンデルブロ集合の画像の生成をする.
    inline void create_images_multi(const std::vector<cv::Vec3b> gradation, int cpu_num) const
    {
        int i = 1;
        T unit = _unit_before;

        _skip_calced_images(unit, i);

        //動画を生成するwhileループ
        while (!_should_end_create_image(unit)) {
            std::vector<std::thread> threads;

            //各スレッドに処理を割り当て
            while (!_should_end_create_image(unit) && threads.size() < cpu_num) {
                //各フレームの画像を生成,保存.
                std::filesystem::path image_path = _calc_image_file_path(i);
                Complex<T> upper_right = _calc_upper_right(unit);

                threads.push_back(std::thread(&MandelbrotMovie<T>::_create_image, this, image_path, upper_right, unit, gradation));

                //次フレームへの処理
                unit *= _change_ratio;
                i++;
            }

            for (int j = 0; j < threads.size(); j++) {
                threads[j].join();
                std::cout << "\r"
                          << "画像の生成" << i - 1 << " / " << _image_qty;
            }
        }
        std::cout << "\r"
                  << "画像の生成が終了しました." << std::endl;
    }

    //生成された画像をもとに動画を生成する.
    void create_movie(std::filesystem::path movie_path, int fourcc, double fps)
    {
        int notify_frequency = 10;  //進捗の通知頻度
        bool isColor = true;        //色付きか否か
        cv::VideoWriter writer(movie_path.string(), fourcc, fps, cv::Size(_width, _height), isColor);

        if (!writer.isOpened()) {
            std::cout << "VideoWriterを開くことができません." << std::endl;
            return;
        }

        //画像をつなげて動画を生成する.
        for (int i = 1; i <= _image_qty; i++) {
            std::filesystem::path image_path = mandelbrot_const::IMAGE_DIR / (std::to_string(i) + ".png");
            cv::Mat image = cv::imread(image_path.string());
            writer << image;
            if (i % notify_frequency == 0) {
                std::cout << "\r"
                          << "動画の生成: " << i << " / " << _image_qty;
            }
        }

        std::cout << "\r"
                  << "動画の生成は終了しました." << std::endl;

        return;
    }

private:
    //各フレームの画像を生成,保存する.
    inline void _create_image(std::filesystem::path image_path, Complex<T> upper_right, T unit, const std::vector<cv::Vec3b> gradation) const
    {
        MandelbrotImage<T> mandelbrot_image(_width, _height, _calc_time, upper_right, unit);
        cv::Mat image = mandelbrot_image.create_image(gradation);
        cv::imwrite(image_path.string(), image);
        return;
    }

    //画像ファイルのパスを返す.
    inline std::filesystem::path _calc_image_file_path(const int& i) const
    {
        std::filesystem::path image_path = mandelbrot_const::IMAGE_DIR / (std::to_string(i) + mandelbrot_const::image_type);
        return image_path;
    }

    //右上の画素に対応する複素数を計算する.
    inline Complex<T> _calc_upper_right(const T& unit) const
    {
        Complex<T> upper_right = _center + Complex<T>(-unit * (_width / 2), unit * (_height / 2));
        return upper_right;
    }

    //動画をつくるのに必要な画像数を計算する.
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

    //十分な画像数か判定する.
    inline bool _should_end_create_image(const T& unit) const
    {
        bool ret = (unit < _unit_after);
        return ret;
    }

    //計算済みの画像の計算をスキップする.
    void _skip_calced_images(T& unit, int& i) const
    {
        T before_unit = unit;
        //計算済みの画像をスキップ.
        while (!_should_end_create_image(unit)) {
            if (std::filesystem::exists(_calc_image_file_path(i))) {
                i++;
                before_unit = unit;
                unit *= _change_ratio;
            } else {
                break;
            }
        }

        //最後に計算された画像は破損している可能性があるため,再計算させる.
        if (i > 1) {
            i--;
            unit = before_unit;
        }
    }

private:
    const Complex<T> _center;  //動画の中心の画素に対応する複素数
    const T _unit_before;      //最初の１画素当たりの辺の長さ
    const T _unit_after;       //最後の１画素当たりの辺の長さ
    const T _change_ratio;     //１フレームごとにunitが何倍になるか
    const int _width;          //動画の幅
    const int _height;         //動画の高さ
    const int _calc_time;      //集合に属するか判定するまでの計算回数
    const int _image_qty;      //動画をつくるのに必要な画像数
};

cv::Vec3b hsv2bgr(double hue, double saturation, double value)
{
    //hsvをbgrに変換する
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
    //色のグラデーションをつくる
    //unit:色相の変化, saturation:彩度, value:明度
    std::vector<cv::Vec3b> v;
    for (double hue = 0; hue < 360; hue += unit) {
        v.push_back(hsv2bgr(hue, saturation, value));
    }
    return v;
}

std::vector<cv::Vec3b> add_repaat(std::vector<cv::Vec3b> v)
{
    //グラデーションを折り返す
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

    //使用するディレクトリの初期化
    init_dirs();
    if (ask_should_execute("計算済みの内容を消去しますか?")) {
        delete_temp_dirs();
        init_dirs();
    }

    //動画内容
    mpf_set_default_prec(1);
    typedef double complex_template_type;                          //complexの実部と虚部に用いる型
    int width = 1920;                                              //動画の幅
    int height = 1080;                                             //動画の高さ
    int calc_time = 1000;                                          //発散を調べる回数
    std::vector<cv::Vec3b> gradation = calc_gradation(1, 50, 70);  //各点の配色
    gradation = add_repaat(gradation);
    Complex<complex_template_type> center(-1.26222162762384535370226702572022420406, -0.04591700163513884695098681782544085357512);  //動画の中心の複素数
    complex_template_type unit_before = 0.001;                                                                                         //動画の最初の縮尺
    complex_template_type unit_after = 0.0000000000000001;                                                               //動画の最後の縮尺
    complex_template_type change_ratio = 0.95;                                                                                       //１フレームごとの縮尺の変化
    std::filesystem::path movie_path = mandelbrot_const::DATA_PATH / "movie2.mp4";                                                   //生成する動画のパス
    const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');                                                                  //コーデック
    double fps = 30.0;                                                                                                               //動画のフレームレート

    MandelbrotMovie mandelbrot_movie(center, unit_before, unit_after, change_ratio, width, height, calc_time);

    //画像の生成
    //mandelbrot_movie.create_images(gradation);
    mandelbrot_movie.create_images_multi(gradation, 10);

    //動画の生成
    mandelbrot_movie.create_movie(movie_path, fourcc, fps);

    return 0;
}