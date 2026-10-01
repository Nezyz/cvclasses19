/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace
{
/// \brief Intensity statistics of a region which can be combined in O(1)
struct region_stats
{
    double count = 0;
    double sum = 0;
    double sum_sq = 0;

    double mean() const
    {
        return sum / count;
    }

    double stddev() const
    {
        // floating point error can make the variance slightly negative for constant regions
        return std::sqrt(std::max(0.0, sum_sq / count - mean() * mean()));
    }
};

region_stats operator+(const region_stats& lhs, const region_stats& rhs)
{
    return {lhs.count + rhs.count, lhs.sum + rhs.sum, lhs.sum_sq + rhs.sum_sq};
}

region_stats compute_stats(const cv::Mat& region)
{
    cv::Scalar mean;
    cv::Scalar dev;
    cv::meanStdDev(region, mean, dev);

    const auto count = static_cast<double>(region.total());
    return {count, count * mean[0], count * (dev[0] * dev[0] + mean[0] * mean[0])};
}

/// \brief Homogeneous rectangle produced by the split stage
struct leaf_region
{
    cv::Rect rect;
    region_stats stats;
};

/// \brief Splits range into two halves, a range of length 1 is left as is
std::vector<cv::Range> halve(const cv::Range& range)
{
    if (range.size() == 1)
        return {range};

    const int middle = range.start + range.size() / 2;
    return {cv::Range(range.start, middle), cv::Range(middle, range.end)};
}

std::vector<cv::Rect> split_in_quarters(const cv::Rect& rect)
{
    std::vector<cv::Rect> parts;
    for (const auto& rows : halve(cv::Range(rect.y, rect.y + rect.height)))
        for (const auto& cols : halve(cv::Range(rect.x, rect.x + rect.width)))
            parts.emplace_back(cols.start, rows.start, cols.size(), rows.size());
    return parts;
}

void split_region(const cv::Mat& image, const cv::Rect& rect, double stddev, std::vector<leaf_region>& leaves)
{
    const auto stats = compute_stats(image(rect));
    if (rect.area() == 1 || stats.stddev() <= stddev)
    {
        leaves.push_back({rect, stats});
        return;
    }

    for (const auto& part : split_in_quarters(rect))
        split_region(image, part, stddev, leaves);
}

/// \brief Disjoint set union of regions which tracks statistics of every set
class region_union
{
    public:
    explicit region_union(const std::vector<leaf_region>& leaves) : parent_(leaves.size())
    {
        std::iota(parent_.begin(), parent_.end(), 0);
        for (const auto& leaf : leaves)
            stats_.push_back(leaf.stats);
    }

    int find(int region)
    {
        // path halving keeps the trees shallow
        while (parent_[region] != region)
        {
            parent_[region] = parent_[parent_[region]];
            region = parent_[region];
        }
        return region;
    }

    /// \brief Merges sets of two regions if their union is still homogeneous
    void merge_if_homogeneous(int lhs, int rhs, double stddev)
    {
        lhs = find(lhs);
        rhs = find(rhs);
        if (lhs == rhs)
            return;

        const auto merged = stats_[lhs] + stats_[rhs];
        if (merged.stddev() > stddev)
            return;

        parent_[rhs] = lhs;
        stats_[lhs] = merged;
    }

    double mean(int region)
    {
        return stats_[find(region)].mean();
    }

    private:
    std::vector<int> parent_;
    std::vector<region_stats> stats_;
};

/// \brief Builds a map where every pixel holds the index of its leaf region
cv::Mat make_label_map(const cv::Size& size, const std::vector<leaf_region>& leaves)
{
    cv::Mat labels(size, CV_32SC1);
    for (int i = 0; i < static_cast<int>(leaves.size()); ++i)
        labels(leaves[i].rect).setTo(i);
    return labels;
}

/// \brief Tries to merge every pair of 4-connected neighbouring regions
void merge_adjacent(const cv::Mat& labels, double stddev, region_union& regions)
{
    for (int y = 0; y < labels.rows; ++y)
    {
        for (int x = 0; x < labels.cols; ++x)
        {
            const int label = labels.at<int>(y, x);
            if (x + 1 < labels.cols && labels.at<int>(y, x + 1) != label)
                regions.merge_if_homogeneous(label, labels.at<int>(y, x + 1), stddev);
            if (y + 1 < labels.rows && labels.at<int>(y + 1, x) != label)
                regions.merge_if_homogeneous(label, labels.at<int>(y + 1, x), stddev);
        }
    }
}

cv::Mat paint_regions(const cv::Size& size, const std::vector<leaf_region>& leaves, region_union& regions)
{
    cv::Mat result(size, CV_64FC1);
    for (int i = 0; i < static_cast<int>(leaves.size()); ++i)
        result(leaves[i].rect).setTo(regions.mean(i));
    return result;
}
} // namespace

namespace cvlib
{
cv::Mat split_and_merge(const cv::Mat& image, double stddev)
{
    // an empty region can not be homogeneous and would be split forever
    CV_Assert(!image.empty());
    // only the first channel statistics are used for homogeneity check
    CV_Assert(image.channels() == 1);

    std::vector<leaf_region> leaves;
    split_region(image, cv::Rect(cv::Point(), image.size()), stddev, leaves);

    region_union regions(leaves);
    merge_adjacent(make_label_map(image.size(), leaves), stddev, regions);

    cv::Mat result;
    paint_regions(image.size(), leaves, regions).convertTo(result, image.type());
    return result;
}
} // namespace cvlib
