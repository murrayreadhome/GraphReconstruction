// C++11
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <set>
#include <string>
#include <fstream>
#include <random>
#include <cmath>
#include <numeric>
#include <deque>
#include <queue>
#include <list>
#include <complex>
#include <iomanip>
#include <iterator>
#include <cstring>
#include <memory>

using namespace std;

#ifdef TESTING_AT_HOME
#ifdef _DEBUG
#define TIME_LIMIT 10000
#define MIN_STEPS 0
#else
#define TIME_LIMIT 1000
#define MIN_STEPS 0
#endif
#else
#define TIME_LIMIT 8000
#define MIN_STEPS 0
#endif

#ifdef TESTING_AT_HOME
#include <time.h>
int milliseconds(bool reset = false)
{
    static clock_t start = clock();
    if (reset)
        start = clock();
    clock_t now = clock();
    clock_t elapsed = now - start;
    elapsed = (elapsed * 1000) / (CLOCKS_PER_SEC);
    return int(elapsed);
}
#else
#include <sys/time.h>
int getTime()
{
    timeval t;
    gettimeofday(&t, NULL);
    int tm = t.tv_sec;
    tm *= 1000;
    tm += (t.tv_usec / 1000);
    return tm;
}
int milliseconds(bool /*reset*/ = false)
{
    static int start = getTime();
    int now = getTime();
    if (now < start)
        now += 60000;	// to account for minute rollover
    int elapsed = now - start;
    return elapsed;
}
#endif

#define ALL(cont) (cont).begin(), (cont).end()

template<bool Condition> struct STATIC_ASSERTION_FAILURE { enum { ASize = -1 }; };
template <> struct STATIC_ASSERTION_FAILURE<true> { enum { ASize = 1 }; };
#define STATIC_ASSERT(cond, name) extern int name[STATIC_ASSERTION_FAILURE<cond>::ASize];

bool keepGoing(int aStart, int aLimit, int aSteps, int aMaxSteps = 2000000000, int aMinSteps = MIN_STEPS)
{
    // max takes precedence over min
    if (aSteps >= aMaxSteps)
        return false;
    if (aSteps < aMinSteps)
        return true;
    int now = milliseconds();
    if (now > aLimit)
        return false;
    int perStep = (now - aStart) / max(aSteps, 1);
    //	cerr << "perstep = " << perStep << " est = " << (aStart + perStep * (aSteps + 1)) << endl;
    return (aStart + perStep * (aSteps + 1)) < aLimit;
}

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
typedef long long LL;

struct Error : public std::exception
{
    string iErr;
    Error(const string& e)
        : iErr(e)
    {}
    ~Error() throw()
    {}
    const char* what() const throw()
    {
        return iErr.c_str();
    }
};

void report(const string& msg)
{
#ifdef TESTING_AT_HOME
    throw Error(msg);
#else
    cerr << msg << endl;
#endif
}

template <typename T>
bool incr(const T& a, const T& b, const T& c)
{
    return a <= b && b <= c;
}
template <typename T>
bool incrStrict(const T& a, const T& b, const T& c)
{
    return a < b && b < c;
}
template <typename T>
bool between(const T& a, const T& b, const T& c)
{
    return min(a, c) <= b && b <= max(a, c);
}
template <typename T>
bool betweenStrict(const T& a, const T& b, const T& c)
{
    return min(a, c) < b && b < max(a, c);
}

const int KUp = -1;
enum TDir { EUp = 0, ERight, EDown, ELeft, ENumDirs, ENorth = EUp, EEast = ERight, ESouth = EDown, EWest = ELeft, ENoDir = 8 };
bool Horizontal(TDir d) { return d == ERight || d == ELeft; }

template <typename T>
struct Point2D
{
    T x;
    T y;
    Point2D()
        : x(), y()
    {}
    template <typename U1, typename U2>
    Point2D(const U1& aX, const U2& aY)
        : x(aX), y(aY)
    {}
    template <typename U>
    Point2D(const Point2D<U> aPoint2D)
        : x(T(aPoint2D.x)), y(T(aPoint2D.y))
    {}
    template <typename U>
    const Point2D& operator=(const Point2D<U> aPoint2D)
    {
        x = T(aPoint2D.x);
        y = T(aPoint2D.y);
        return *this;
    }
    template <typename U>
    Point2D operator+(const Point2D<U>& aPoint2D) const
    {
        return Point2D(x + aPoint2D.x, y + aPoint2D.y);
    }
    template <typename U>
    const Point2D& operator+=(const Point2D<U>& aPoint2D)
    {
        x += aPoint2D.x;
        y += aPoint2D.y;
        return *this;
    }
    Point2D operator-(const Point2D& aPoint2D) const
    {
        return Point2D(x - aPoint2D.x, y - aPoint2D.y);
    }
    const Point2D& operator-=(const Point2D& aPoint2D)
    {
        x -= aPoint2D.x;
        y -= aPoint2D.y;
        return *this;
    }
    bool operator==(const Point2D& aPoint2D) const
    {
        return x == aPoint2D.x && y == aPoint2D.y;
    }
    bool operator!=(const Point2D& aPoint2D) const
    {
        return x != aPoint2D.x || y != aPoint2D.y;
    }
    Point2D operator*(T aFactor) const
    {
        return Point2D(x * aFactor, y * aFactor);
    }
    const Point2D& operator*=(T aFactor)
    {
        x = x * aFactor;
        y = y * aFactor;
        return *this;
    }
    Point2D operator/(T aFactor) const
    {
        return Point2D(x / aFactor, y / aFactor);
    }
    const Point2D& operator/=(T aFactor)
    {
        x = x / aFactor;
        y = y / aFactor;
        return *this;
    }
    Point2D operator-() const
    {
        return Point2D(-x, -y);
    }
    bool operator<(const Point2D& aOther) const
    {
        return (y < aOther.y) || ((y == aOther.y) && (x < aOther.x));
    }
    T cross(const Point2D& aP) const
    {
        return x * aP.y - y * aP.x;
    }
    T operator*(const Point2D& aP) const
    {
        return cross(aP);
    }
    T dot(const Point2D& aP) const
    {
        return x * aP.x + y * aP.y;
    }
    T operator()(const Point2D& aP) const
    {
        return dot(aP);
    }
    T boxDist(const Point2D& aPoint2D) const
    {
        return abs(x - aPoint2D.x) + abs(y - aPoint2D.y);
    }
    T squaredMagnitude() const
    {
        return x * x + y * y;
    }
    double magnitude() const
    {
        return sqrt(double(squaredMagnitude()));
    }
    double dist(const Point2D& aPoint2D) const
    {
        T dx = x - aPoint2D.x;
        T dy = y - aPoint2D.y;
        return sqrt(double(dx * dx + dy * dy));
    }
    double angle()
    {
        return atan2(double(y), double(x));
    }
    Point2D<double> unit() const
    {
        double mag = magnitude();
        if (mag != 0.0)
            return Point2D<double>(x / mag, y / mag);
        else
            return Point2D<double>(0.0, 0.0);
    }
    Point2D rotate(double aAngle) const
    {
        double c = cos(aAngle);
        double s = sin(aAngle);
        return Point2D(T(x * c - y * s), T(x * s + y * c));
    }
    Point2D rotateCW() const
    {
        return Point2D(-y, x);
    }
    Point2D rotateCCW() const
    {
        return Point2D(y, -x);
    }
    Point2D swapped() const
    {
        return Point2D(y, x);
    }
    T operator[](TDir dir) const
    {
        switch (dir)
        {
        case EUp:
        case EDown:
            return y;
        case ELeft:
        case ERight:
            return x;
        default:
            return T();
        }
    }
    T& operator[](TDir dir)
    {
        switch (dir)
        {
        case EUp:
        case EDown:
            return y;
        case ELeft:
        case ERight:
        default:
            return x;
        }
    }
};

template <typename T>
Point2D<T> minPos(const Point2D<T>& a, const Point2D<T>& b)
{
    return Point2D<T>(min(a.x, b.x), min(a.y, b.y));
}

template <typename T>
Point2D<T> maxPos(const Point2D<T>& a, const Point2D<T>& b)
{
    return Point2D<T>(max(a.x, b.x), max(a.y, b.y));
}

template <typename T>
ostream& operator<<(ostream& aStream, const Point2D<T>& aPoint2D)
{
    aStream << aPoint2D.x << "," << aPoint2D.y;
    return aStream;
}

typedef Point2D<int> Pos;
typedef Point2D<LL> PosLL;
typedef Point2D<double> Point;
typedef Point2D<float> PointF;
template <typename T>
Pos toPos(const Point2D<T>& p)
{
    return Pos(int(p.x + 0.5), int(p.y + 0.5));
}

const double triEps = 1e-12;
template <typename T>
bool pointInTriangle(const Point2D<T>& p, const Point2D<T>& a, const Point2D<T>& b, const Point2D<T>& c)
{
    double x = (a - b) * (a - p), y = (b - c) * (b - p), z = (c - a) * (c - p);
    return ((x <= triEps && y <= triEps && z <= triEps) || (x >= -triEps && y >= -triEps && z >= -triEps));
}

const Pos KDir[9] = { Pos(0,KUp), Pos(1,0), Pos(0,-KUp), Pos(-1,0), Pos(1,-KUp), Pos(1,KUp), Pos(-1,-KUp), Pos(-1,KUp), Pos(0,0) };
const string KDirName = "URDL";
const string KCompassName = "NESW";
const TDir KOppositeDir[4] = { EDown, ELeft, EUp, ERight };

TDir dirTo(const Pos& aFrom, const Pos& aTo)
{
    int dx = aTo.x - aFrom.x;
    int dy = aTo.y - aFrom.y;
    //	if (dx && dy)
    //		throw Error("No Direction!");
    if (abs(dy) > abs(dx))
    {
        bool b = dy < 0;
#ifdef TESTING_AT_HOME
#pragma warning(disable:4127)
#endif
        if (KUp < 0)
            b = !b;
#ifdef TESTING_AT_HOME
#pragma warning(default:4127)
#endif
        return b ? EDown : EUp;
    }
    else
        return dx > 0 ? ERight : (dx < 0 ? ELeft : ENoDir);
}

struct TDirs { TDir x, y; };
TDirs dirsTo(const Pos& aFrom, const Pos& aTo)
{
    TDirs dirs = { ENoDir, ENoDir };
    if (aTo.x < aFrom.x)
        dirs.x = ELeft;
    if (aTo.x > aFrom.x)
        dirs.x = ERight;
    if (aTo.y < aFrom.y)
        dirs.y = EUp;
    if (aTo.y > aFrom.y)
        dirs.y = EDown;
#if	KUp == 1
    dirs.second = TDir(EDown - dirs.second);
#endif
    return dirs;
}

template <typename T>
struct Grid
{
    typedef vector<T> Coll;
    typedef typename Coll::iterator iterator;
    typedef typename Coll::const_iterator const_iterator;
    typedef typename Coll::reference Ref;
    typedef typename Coll::const_reference ConstRef;

    Grid()
        : width(0), height(0), bWidth(0), bHeight(0), border(0)
    {}

    Grid(int aWidth, int aHeight, const T& aVal)
    {
        init(aWidth, aHeight, aVal, 0);
    }

    Grid(int aWidth, int aHeight, const T& aVal, int aBorder)
    {
        init(aWidth, aHeight, aVal, aBorder);
    }

    Grid(const Grid& aGrid)
    {
        *this = aGrid;
    }

    const Grid& operator=(const Grid& aGrid)
    {
        if (&aGrid == this)
            return *this;
        width = aGrid.width;
        height = aGrid.height;
        bWidth = aGrid.bWidth;
        bHeight = aGrid.bHeight;
        border = aGrid.border;
        iGrid = aGrid.iGrid;
        iBegin = iGrid.begin();
        iBegin = iter(Pos(border, border));
        return *this;
    }

    void init(int aWidth, int aHeight, const T& aVal, int aBorder = 0)
    {
        bWidth = aWidth + aBorder * 2;
        bHeight = aHeight + aBorder * 2;
        width = aWidth;
        height = aHeight;
        border = aBorder;
        iGrid.clear();
        iGrid.resize(bWidth * bHeight, aVal);
        iBegin = iGrid.begin();
        iBegin = iter(Pos(aBorder, aBorder));
    }

    inline int index(const Pos& aPos) const
    {
#ifdef _DEBUG
        size_t base = border * (bWidth + 1);
        size_t idx = aPos.x + aPos.y * bWidth + base;
        if (idx < 0 || idx > size(iGrid))	// allow for "end" positions
            report("Grid bad pos");
        return int(idx - base);
#else
        return aPos.x + aPos.y * bWidth;
#endif
    }

    Ref operator[](const Pos& aPos)
    {
        return iBegin[index(aPos)];
    }

    T operator[](const Pos& aPos) const
    {
        return iBegin[index(aPos)];
    }

    ConstRef get(const Pos& aPos) const
    {
        return iBegin[index(aPos)];
    }

    Ref get(const Pos& aPos)
    {
        return iBegin[index(aPos)];
    }

    void set(const Pos& aPos, const T& v)
    {
        iBegin[index(aPos)] = v;
    }

    iterator begin()
    {
        return iGrid.begin();
    }

    iterator end()
    {
        return iGrid.end();
    }

    const_iterator begin() const
    {
        return iGrid.begin();
    }

    const_iterator end() const
    {
        return iGrid.end();
    }

    iterator iter(const Pos& aPos)
    {
        return iBegin + index(aPos);
    }

    const_iterator iter(const Pos& aPos) const
    {
        return iBegin + index(aPos);
    }

    bool isValid(const Pos& aPos) const
    {
        return 0 <= aPos.x && aPos.x < width && 0 <= aPos.y && aPos.y < height;
    }

    bool isBorderValid(const Pos& aPos) const
    {
        return -border <= aPos.x && aPos.x < width + border && -border <= aPos.y && aPos.y < height + border;
    }

    void limit(Pos& aPos) const
    {
        aPos.x = min(max(0, aPos.x), width - 1);
        aPos.y = min(max(0, aPos.y), height - 1);
    }

    template<typename T2>
    void swap(Grid<T2>& other)
    {
        iGrid.swap(other.iGrid);
        swap(width, other.width);
        swap(height, other.height);
        swap(bWidth, other.bWidth);
        swap(bHeight, other.bHeight);
        swap(border, other.border);
        swap(iBegin, other.iBegin);
    }

    int width;
    int height;
    int bWidth;
    int bHeight;
    int border;
    Coll iGrid;
    typename Coll::iterator iBegin;
};

#define FOR_GRID(var, grid)    for (Pos var(0,0); var.y<grid.height; var.y++) for (var.x=0; var.x<grid.width; var.x++)
#define FOR_SIZE(var, w, h)    for (Pos var(0,0); var.y<(h); var.y++) for (var.x=0; var.x<(w); var.x++)
#define FOR_RECT(var, l, t, r, b)    for (Pos var((l),(t)); var.y<(b); var.y++) for (var.x=(l); var.x<(r); var.x++)

template<typename T>
ostream& operator<<(ostream& aStream, const Grid<T>& aGrid)
{
    for (Pos var(0, 0); var.y < aGrid.height; var.y++)
    {
        for (var.x = 0; var.x < aGrid.width; var.x++)
        {
            aStream << aGrid[var] << ",";
        }
        aStream << endl;
    }
    return aStream;
}

template<typename T>
int Size(const T& cont)
{
    return int(cont.size());
}

template<typename T>
ostream& operator<<(ostream& aStream, const vector<T>& aVect)
{
    int s = Size(aVect);
    aStream << "{";
    for (int i = 0; i < s; i++)
    {
        if (i > 0)
            aStream << ",";
        aStream << aVect[i];
    }
    aStream << "}" << endl;
    return aStream;
}

const double PI = acos(-1.0);
const double PI2 = PI * 2;
const double SQRT2 = sqrt(2.0);

inline double angleNorm(double a)
{
    if (a < -PI)
    {
        int times = int((a - PI) / PI2);
        a -= times * PI2;
    }
    else if (a > PI)
    {
        int times = int((a + PI) / PI2);
        a -= times * PI2;
    }
    return a;
}
inline double angleAdd(double a, double b)
{
    return angleNorm(a + b);
}
inline double angleSub(double a, double b)
{
    return angleNorm(a - b);
}

struct OrderPointByAngleToRef
{
    Point iRef;
    OrderPointByAngleToRef(const Point& aRef) : iRef(aRef)
    {}
    bool operator()(const Point& aL, const Point& aR) const
    {
        return (aL - iRef).angle() > (aR - iRef).angle();
    }
};

struct OrderPointByDistToRef
{
    Point iRef;
    OrderPointByDistToRef(const Point& aRef) : iRef(aRef)
    {}
    bool operator()(const Point& aL, const Point& aR) const
    {
        return (aL - iRef).squaredMagnitude() < (aR - iRef).squaredMagnitude();
    }
};

uint hash(const string& str)
{
    uint hash = 5381;
    for (int i = 0; i < Size(str); i++)
        hash = hash * 33 ^ str[i];
    return hash;
}

int bitCount[256] =
{
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};

inline int countBits(int bits)
{
    unsigned char* bytes = (unsigned char*)&bits;
    return bitCount[bytes[0]] + bitCount[bytes[1]] + bitCount[bytes[2]] + bitCount[bytes[3]];
}

template<typename TDATA, typename TSCORE>
class TrackMin
{
public:
    TDATA data;
    TSCORE val;
    TrackMin(const TDATA& startData, const TSCORE& startVal)
        : data(startData), val(startVal)
    {}

    bool update(const TDATA& newData, const TSCORE& newVal)
    {
        if (newVal < val)
        {
            data = newData;
            val = newVal;
            return true;
        }
        return false;
    }
};

template<typename TDATA, typename TSCORE>
class TrackMax
{
public:
    TDATA data;
    TSCORE val;
    TrackMax(const TDATA& startData, const TSCORE& startVal)
        : data(startData), val(startVal)
    {}

    bool update(const TDATA& newData, const TSCORE& newVal)
    {
        if (newVal > val)
        {
            data = newData;
            val = newVal;
            return true;
        }
        return false;
    }
};

template<typename TDATA, typename TSCORE>
class TrackMaxEq
{
public:
    TDATA data;
    TSCORE val;
    TrackMaxEq(const TDATA& startData, const TSCORE& startVal)
        : data(startData), val(startVal)
    {}

    bool update(const TDATA& newData, const TSCORE& newVal)
    {
        if (newVal >= val)
        {
            data = newData;
            val = newVal;
            return true;
        }
        return false;
    }
};

template <typename T, typename U>
U& other(const T& from, U& a, U& b)
{
    if (from == a)
        return b;
    else if (from == b)
        return a;
    else
        report("neither");
    return a;
}

#ifdef TESTING_AT_HOME
inline bool isnan(double x)
{
    return x != x;
}
#endif

////////////////////////////////////////////////////////
////////////////  Solution starts here  ////////////////
////////////////////////////////////////////////////////

struct Problem
{
    int N;
    double C;
    int K;
    Grid<bool> AdjMat;
    vector<string> Paths;

    Problem(int seed)
    {
        // generate test case
        mt19937 re(seed);
  
        //generate number of nodes
        N = uniform_int_distribution<int>(10, 100)(re);
        
        //set easiest/hardest seeds
        if (seed == 1) N = 10;
        if (seed == 2) N = 100;
        
        //generate connectivity value C
        C = uniform_real_distribution<double>(1.0/N, 3.0/N)(re);

        //generate K parameter
        K = uniform_int_distribution<int>(1, 10)(re);
        if (seed == 1) K = 2;
        if (seed == 2) K = 10;


        //generate the true adjacency matrix
        AdjMat.init(N, N, false);

        for (int i=0; i<N; i++)
          for (int k=i+1; k<N; k++)
            if (uniform_real_distribution<double>(0.0, 1.0)(re) < C)
            {
              AdjMat[Pos(i,k)]=true;
              AdjMat[Pos(k,i)]=true;
            }

        //compute shortest paths between every pair of nodes
        int Inf = 1000000;
        Grid<int> shortestPath(N,N,0);
        for (int i=0; i<N; i++)
        {
          shortestPath[Pos(i,i)]=0;
          for (int k=i+1; k<N; k++)
          {
            if (AdjMat[Pos(i,k)]) shortestPath[Pos(i,k)]=1;
            else  shortestPath[Pos(i,k)]=Inf;

            shortestPath[Pos(k,i)]=shortestPath[Pos(i,k)];      //make symmetric
          }
        }

        //run all-pairs shortest path. Best algorithm ever!
        for (int k=0; k<N; k++)
          for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
              shortestPath[Pos(i,j)]=min(shortestPath[Pos(i,j)],shortestPath[Pos(i,k)]+shortestPath[Pos(k,j)]);        
              
        //convert Inf to -1
        for (int i=0; i<N; i++)
          for (int k=0; k<N; k++)
            if (shortestPath[Pos(i,k)]==Inf)
              shortestPath[Pos(i,k)]=-1;

        //generate paths through the graph
        vector<int> seen(N,0);
        Grid<bool> seen2(N,N,false);
        int countSeen=0;

        vector<int> ind;
        for (int i=0; i<N*N; i++) ind.push_back(i);
        shuffle(ind.begin(), ind.end(), re);

        for (int i=0; i<int(ind.size()); i++)
        {
          int node1=ind[i]/N;
          int node2=ind[i]%N;
          if (node1>=node2) continue;           //out of order
          if (seen2[Pos(node1,node2)]) continue;    //already seen this pair

          seen[node1]++;
          if (seen[node1]==K) countSeen++;
          seen[node2]++;
          if (seen[node2]==K) countSeen++;
          seen2[Pos(node1,node2)]=true;     
          seen2[Pos(node2,node1)]=true;
          stringstream path;
          path << node1 << " " << node2 << " " << shortestPath[Pos(node1,node2)];
          Paths.push_back(path.str());

          if (countSeen==N) break;      //seen all nodes at least K times         
        }        
    }

    double score(const vector<string>& AdjPredicted) const
    {
                  //compute the raw score 
        double TP=0;         //number of green edges
        double FP=0;         //number of red edges
        double FN=0;         //number of true edges we missed
        double TN=0;         //number of non-edges we missed correctly

        for (int r=0; r<N; r++)
            for (int c=r+1; c<N; c++)
            {
                if (AdjPredicted[r][c]=='1')
                {
                    if (AdjMat[Pos(r,c)]) TP++;
                    else FP++;
                }
                else
                {
                    if (AdjMat[Pos(r,c)]) FN++;
                    else TN++;
                }
            }

        double Precision = (TP+FP==0 ? 0 : TP*1.0/(TP+FP));
        double Recall = (TP+FN==0 ? 0 : TP*1.0/(TP+FN));
        //compute the F1 score
        double Score = (Precision + Recall < 1e-9 ? 0 : (2 * Precision * Recall)/(Precision + Recall));

        cout << "TN=" << TN << " FP=" << FP << " FN=" << FN << " TP=" << TP << " ";

        return Score;
    }
};

uniform_real_distribution<double>r01(0.0, 1.0);

class GraphReconstruction 
{
public:
    struct PathDist
    {
        size_t from;
        size_t to;
        int dist;
    };

    vector<PathDist> path_dists;
    size_t N;
    double C;
    size_t K;

    mt19937 re;

    struct EdgeDist
    {
        size_t to;
        int dist;
    };
    vector<vector<EdgeDist>> connections;
    Grid<bool> disconnected;

    vector<string> findSolution(int aN, double aC, int aK, const vector<string>& paths)
    {
        milliseconds(true);

        N = aN;
        C = aC;
        K = aK;

        path_dists.reserve(paths.size());
        for (const string& path : paths)
        {
            stringstream strm(path);
            PathDist pd;
            strm >> pd.from >> pd.to >> pd.dist;
            path_dists.push_back(pd);
        }

        make_islands();
        minimum_distances();
        optimise_islands();

        vector<string> out;

        for (size_t i=0; i<N; i++)
        {
            string row;
            for (size_t k=0; k<N; k++) row+= disconnected[Pos(i,k)] ? "0" : "1";
            out.push_back(row);
        }

        return out;
    }
    
    struct Island
    {
        vector<size_t> nodes;
        size_t min_size;
        vector<PathDist> paths;
    };
    vector<Island> islands;

    void make_islands()
    {
        // initial connection information
        connections.resize(N);
        disconnected.init(N, N, false);
        for (size_t i = 0; i < N; i++)
            disconnected[Pos(i, i)] = true;
        for (const PathDist& pd : path_dists)
        {
            if (pd.dist == -1)
            {
                disconnect(Pos(pd.from, pd.to), true);
            }
            else
            {
                connections[pd.from].push_back({ pd.to, pd.dist });
                connections[pd.to].push_back({ pd.from, pd.dist });
            }
        }

        // islands
        vector<int> seen(N, 0);
        for (size_t i = 0; i < N; i++)
        {
            if (seen[i])
                continue;
            Island island{ {i}, 0 };
            seen[i] = 1;
            deque<size_t> queue(1, i);
            while (!queue.empty())
            {
                size_t n = queue.front();
                queue.pop_front();
                for (const EdgeDist& ed : connections[n])
                {
                    island.min_size = max(island.min_size, size_t(ed.dist + 1));
                    if (seen[ed.to])
                        continue;
                    seen[ed.to] = 1;
                    island.nodes.push_back(ed.to);
                    queue.push_back(ed.to);
                }
            }
            islands.push_back(island);
        }

        // disconnect definitely disconnected islands
        size_t ni = islands.size();
        for (size_t i = 0; i < ni - 1; i++)
        {
            for (size_t j = i + 1; j < ni; j++)
            {
                if (islands_disconnected(islands[i], islands[j]))
                {
                    disconnect_islands(islands[i], islands[j]);
                }
            }
        }

        // join islands that are too small
        bool joined = true;
        while (joined)
        {
            joined = false;
            size_t ni = islands.size();
            for (size_t i = 0; i < ni; i++)
            {
                Island& a = islands[i];
                if (a.nodes.size() >= a.min_size)
                    continue;
                size_t bestj = N;
                size_t best_extra = N;
                size_t needed = a.min_size - a.nodes.size();
                for (size_t j = 0; j < ni; j++)
                {
                    if (i == j)
                        continue;
                    Island& b = islands[j];
                    if (islands_disconnected(a, b))
                        continue;
                    size_t extra = b.nodes.size();
                    if (extra < best_extra || (extra <= needed && extra > best_extra))
                    {
                        best_extra = extra;
                        bestj = j;
                    }
                }
                if (bestj == N)
                    continue;
                joined = true;
                Island& b = islands[bestj];
                a.nodes.insert(a.nodes.end(), b.nodes.begin(), b.nodes.end());
                a.min_size = max(a.min_size, b.min_size);
                swap(b, islands.back());
                islands.pop_back();
                break;
            }
        }

        // disconnect all remaining islands
        ni = islands.size();
        for (size_t i = 0; i < ni - 1; i++)
        {
            for (size_t j = i + 1; j < ni; j++)
            {
                disconnect_islands(islands[i], islands[j]);
            }
        }

        // Add the relevant paths
        for (Island& island : islands)
        {
            sort(island.nodes.begin(), island.nodes.end());
            for (size_t n : island.nodes)
            {
                for (const EdgeDist& ed : connections[n])
                {
                    if (n>ed.to || ed.dist==-1)
                        continue;
                    island.paths.push_back({n, ed.to, ed.dist});
                }
            }
        }
    }

    bool islands_disconnected(const Island& a, const Island& b) const
    {
        for (size_t i : a.nodes)
            for (size_t j : b.nodes)
                if (disconnected[Pos(i, j)])
                    return true;
        return false;
    }

    void disconnect_islands(const Island& a, const Island& b)
    {
        for (size_t i : a.nodes)
        {
            for (size_t j : b.nodes)
            {
                disconnect(Pos(i, j), true);
            }
        }
    }

    Grid<int> min_dist;
    void minimum_distances()
    {
        min_dist.init(N,N,1);
        for (const Island& island : islands)
        {
            for (size_t n : island.nodes)
            {
                for (const EdgeDist& ed : connections[n])
                {
                    if (ed.dist <= 1)
                        continue;
                    disconnect(Pos(n, ed.to), true);
                    propagate_min_dist(n, ed.to, ed.dist);
                }
            }
        }
    }

    void propagate_min_dist(size_t from, size_t to, int dist)
    {
        deque<EdgeDist> queue(1, EdgeDist{from, 0});
        while (!queue.empty())
        {
            EdgeDist ed = queue.front();
            size_t f = ed.to;
            int d = ed.dist;
            queue.pop_front();
            for (const EdgeDist& edge : connections[f])
            {
                int remaining = d - edge.dist;
                if (remaining <= 1)
                    continue;
                disconnect(Pos(f, to), true);
                int& mda = min_dist[Pos(f, to)];
                int& mdb = min_dist[Pos(to, f)];
                if (remaining > mda)
                {
                    mda = remaining;
                    mdb = remaining;
                }
                queue.push_back({edge.to, remaining});
            }
        }
        // todo add these soft constraints to islands
    }

    void optimise_islands()
    {
        vector<double> island_rel;
        for (Island& island : islands)
            island_rel.push_back(pow(double(island.nodes.size()), 4));
        double tot = accumulate(ALL(island_rel), 0.0);
        for (size_t i=0; i<islands.size(); i++)
        {
            if (i > 0)
                island_rel[i] += island_rel[i - 1];
            const Island& island = islands[i];
            optimise_island_1(island);
            optimise_island(island, int(TIME_LIMIT * island_rel[i] / tot));
        }
    }
    
    struct Change
    {
        Pos added;
        Pos removed;
    };

    const Pos no_pos = { 1000, 1000 };

    void optimise_island(const Island& island, int time_limit)
    {
        vector<Pos> present;
        vector<Pos> removed;
        for (size_t i : island.nodes)
        {
            for (size_t j : island.nodes)
            {
                Pos p(i, j);
                if (i < j && min_dist[p] == 1)
                {
                    if (disconnected[p])
                        removed.push_back(p);
                    else
                        present.push_back(p);
                }
            }
        }
        // mutate by moving edged between present and removed
        
        vector<Pos> best_present = present;
        vector<Pos> best_removed = removed;
        double best_score = score_island(island);

        size_t M = island.nodes.size();
        while (best_score > 0 && milliseconds() < time_limit)
        {
            bool random = re() % 2;
            Change change;
            if (random)
                change = random_mutation(present, removed);
            else
                change = targetted_change(island, present, removed);
            double score = score_island(island);
            if (score <= best_score)
            {
                best_score = score;
                best_present = present;
                best_removed = removed;
            }
            else
                undo(change, present, removed);
        }
        //cout << best_score << " " << island.nodes.size() << endl;

        apply_present(island, best_present, best_removed);
    }

    Change targetted_change(const Island& island, vector<Pos>& present, vector<Pos>& removed)
    {
        calc_apsp(island);
        PathDist path_dist;
        int bad_count = 0;
        for (const PathDist& pd : island.paths)
        {
            int sp = apsp[Pos(pd.from, pd.to)];
            if (sp != pd.dist)
            {
                bad_count++;
                if (r01(re) < 1.0 / bad_count)
                    path_dist = pd;
            }
        }

        if (!bad_count)
            return { no_pos, no_pos };

        if (apsp[Pos(path_dist.from, path_dist.to)] > path_dist.dist)
            return shorten(island, path_dist, present, removed);
        else
            return lengthen(island, path_dist, present, removed);
    }

    Change shorten(const Island& island, const PathDist& path_dist, vector<Pos>& present, vector<Pos>& removed)
    {
        vector<PathDist> from_steps;
        get_steps(island, path_dist.from, path_dist.to, from_steps);
        vector<PathDist> to_steps;
        get_steps(island, path_dist.to, path_dist.from, to_steps);
        // pick a random join that sets the right length

        int count = 0;
        int best_err = 1000;
        Pos to_add;
        for (const PathDist& from_step : from_steps)
        {
            for (const PathDist& to_step : to_steps)
            {
                Pos p{ from_step.to, to_step.to };
                if (min_dist[p] > 1)
                    continue;
                int join_dist = from_step.dist + to_step.dist + 1;
                int err = abs(join_dist - path_dist.dist);
                if (err < best_err)
                {
                    best_err = err;
                    count = 0;
                }
                if (err == best_err)
                {
                    count++;
                    if (r01(re) < 1.0 / count)
                        to_add = p;
                }
            }
        }
        if (!count)
            return { no_pos, no_pos };
        if (to_add.x > to_add.y)
            to_add = to_add.swapped();
        auto where = find(ALL(removed), to_add);
        swap(*where, removed.back());
        removed.pop_back();
        present.push_back(to_add);
        disconnect(to_add, false);
        return { to_add, no_pos };
    }

    Change lengthen(const Island& island, const PathDist& path_dist, vector<Pos>& present, vector<Pos>& removed)
    {
        vector<PathDist> steps;
        get_steps(island, path_dist.from, path_dist.to, steps);
        // pick a random path element to remove
        PathDist to_remove;
        int count = 0;
        for (const PathDist& step : steps)
        {
            if (step.dist <= path_dist.dist && step.dist != 0)
            {
                count++;
                if (r01(re) < 1.0 / count)
                    to_remove = step;
            }
        }
        Pos r{ to_remove.from, to_remove.to };
        if (r.x > r.y)
            r = r.swapped();
        auto where = find(ALL(present), r);
        swap(*where, present.back());
        present.pop_back();
        removed.push_back(r);
        disconnect(r, true);
        return { no_pos, r };
    }

    void get_steps(const Island& island, size_t from, size_t to, vector<PathDist>& steps)
    {
        steps.clear();
        steps.push_back({ from, from, 0 });
        vector<int> seen(N, N);
        seen[from] = from;
        vector<size_t> now, next;
        now.push_back(from);
        int dist = 1;
        while (!now.empty())
        {
            next.clear();
            for (size_t f : now)
            {
                for (size_t to = 0; to < N; to++)
                {
                    if (seen[to] < dist)
                        continue;
                    if (disconnected[{from, to}])
                        continue;
                    steps.push_back({ f,to,dist });
                    if (seen[to] > dist)
                        next.push_back(to);
                    seen[to] = dist;
                }
            }
            dist++;
            now.swap(next);
        }
    }

    double score_island(const Island& island)
    {
        calc_apsp(island);
        double score = 0;
        for (const PathDist& pd : island.paths)
        {
            int sp = apsp[Pos(pd.from, pd.to)];
            int err = sp - pd.dist;
            score += err * err;
        }
        return score;
    }

    Change random_mutation(vector<Pos>& present, vector<Pos>& removed)
    {
        Change change{ no_pos, no_pos };

        auto remove_random = [this](vector<Pos>& v)
        {
            if (v.empty())
                return no_pos;
            swap(v[re() % v.size()], v.back());
            Pos p = v.back();
            v.pop_back();
            return p;
        };

        switch (re() % 3)
        {
        case 0:
            change.removed = remove_random(present);
            break;
        case 1:
            change.added = remove_random(removed);
            break;
        case 2:
            change.removed = remove_random(present);
            change.added = remove_random(removed);
            break;
        }

        if (change.added != no_pos)
        {
            present.push_back(change.added);
            disconnect(change.added, false);
        }
        if (change.removed != no_pos)
        {
            removed.push_back(change.removed);
            disconnect(change.removed, true);
        }

        return change;
    }

    void disconnect(const Pos& p, bool disconnection)
    {
        disconnected[p] = disconnection;
        disconnected[p.swapped()] = disconnection;
    }

    void undo(const Change& change, vector<Pos>& present, vector<Pos>& removed)
    {
        if (change.added != no_pos)
        {
            if (change.added != present.back())
                report("undo remove fail");
            present.pop_back();
        }
        if (change.removed != no_pos)
        {
            if (change.removed != removed.back())
                report("undo remove fail");
            removed.pop_back();
            present.push_back(change.removed);
            disconnect(change.removed, false);
        }
        if (change.added != no_pos)
        {
            removed.push_back(change.added);
            disconnect(change.added, true);
        }
    }

    void apply_present(const Island& island, const vector<Pos>& present, const vector<Pos>& removed)
    {
        for (const Pos& p : present)
            disconnect(p, false);
        for (const Pos& p : removed)
            disconnect(p, true);
    }


    void optimise_island_1(const Island& island)
    {
        // initially all nodes are connected, except those we know can't be
        // how to optimise?
        // eliminate edges randomly
        // track which ones break the rules (cause disconnect or greater path)
        // stop when all rules are matched exactly, or no more edges can be removed
        // redo until timeout, keep best
        // this never satisfies a non trivial rules set
        // COULD MAYBE... evaluate how badly the constraints are broken, and SA to improve that down to zero?
        size_t M = island.nodes.size();
        vector<Pos> removable;
        for (size_t i : island.nodes)
            for (size_t j : island.nodes)
                if (i<j && !disconnected[Pos(i,j)])
                    removable.push_back(Pos(i,j));
        // todo track what was disconnected and restore it for a retry
        // totry identify troublesome edges and leave the rest alone? 
        shuffle(removable.begin(), removable.end(), re);
        bool rules_satisfied = false;
        bool ok_at_start = check_constraints(island, rules_satisfied);
        if (!ok_at_start) cerr << "not ok at start??!" << endl;
        while (!removable.empty() && !rules_satisfied)
        {
            Pos remove = removable.back();
            removable.pop_back();
            disconnect(remove, true);
            bool ok = check_constraints(island, rules_satisfied);
            if (!ok)
            {
                rules_satisfied = false;
                disconnect(remove, false);
            }
        }
        //cerr << rules_satisfied << " " << island.nodes.size() << endl;
    }

    Grid<int> apsp;  // todo optimise

    void calc_apsp(const Island& island)
    {
        apsp.init(N, N, island.nodes.size()+1);  // todo optimise

        vector<vector<size_t>> edges(N);
        for (size_t i : island.nodes)
            for (size_t j : island.nodes)
                if (!disconnected[{i, j}])
                    edges[i].push_back(j);

        vector<int> seen;
        vector<size_t> now, next;
        for (size_t i : island.nodes)
        {
            seen.clear();
            seen.resize(N, 0);
            seen[i] = 1;
            now.clear();
            now.push_back(i);
            int dist = 1;
            while (!now.empty())
            {
                next.clear();
                for (size_t j : now)
                {
                    for (size_t k : edges[j])
                    {
                        if (seen[k])
                            continue;
                        seen[k] = 1;
                        next.push_back(k);
                        apsp[{i, k}] = dist;
                    }
                }
                dist++;
                now.swap(next);
            }
        }
    }

    bool check_constraints(const Island& island, bool& satisfied)
    {
        calc_apsp(island);

        satisfied = true;
        for (const PathDist& pd : island.paths)
        {
            int sp = apsp[Pos(pd.from, pd.to)];
            if (sp != pd.dist)
                satisfied = false;
            if (sp > pd.dist)
                return false;
        }

        return true;
    }
};

vector<string> run(const Problem& prob)
{
    GraphReconstruction prog;
    return prog.findSolution(prob.N, prob.C, prob.K, prob.Paths);
}

double eval(const Problem& prob)
{
    vector<string> pred = run(prob);
    return prob.score(pred);
}

double eval(int n)
{
    return eval(Problem(n));
}

void local()
{
    for (int i=0; i<10; i++)
        cout << eval(i) << endl;
}

void stdio_main() {
    GraphReconstruction prog;
    int N;
    double C;
    int K;
    int NumPaths;
    vector<string> paths;
    string path;
    cin >> N;
    cin >> C;
    cin >> K;
    cin >> NumPaths;
    getline(cin, path);
    for (int i=0; i<NumPaths; i++)
    {
        getline(cin, path);
        paths.push_back(path);
    }
    
    vector<string> ret = prog.findSolution(N,C,K,paths);
    cout << ret.size() << endl;
    for (int i = 0; i < (int)ret.size(); ++i)
            cout << ret[i] << endl;
    cout.flush();
}

void record(int argc, char** argv)
{
    ofstream out(argv[0]);
    while (!cin.eof())
    {
        char ch;
        cin >> ch;
        out << ch;
    }
}

int main(int argc, char** argv)
{
#ifdef TESTING_AT_HOME
    string mode = "local";
#else
    string mode = "stdio";
#endif
    if (argc > 1)
    {
        mode = argv[1];
        argc += 2;
        argv += 2;
    }
    if (mode == "stdio")
        stdio_main();
    else if (mode == "local")
        local();
    else if (mode == "record")
        record(argc, argv);
    else
    {
        cout << "unrecognised mode " << mode << endl;
    }
    return 0;
}
