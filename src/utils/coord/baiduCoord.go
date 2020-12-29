package main

import (
	"C"

	"github.com/spencer404/go-bd09mc"
)

//export LL2MC
func LL2MC(lng, lat float64) (float64, float64) {
	var _ error
	lng, lat, _ = bd09mc.LL2MC(lng, lat)
	return lng, lat
}

//export MC2LL
func MC2LL(lng, lat float64) (float64, float64) {
	var _ error
	lng, lat, _ = bd09mc.MC2LL(lng, lat)
	// fmt.Println(lng, lat, err)
	return lng, lat
}

//export LL2MC_lng
func LL2MC_lng(lng, lat float64) float64 {
	var _ error
	lng, lat, _ = bd09mc.LL2MC(lng, lat)
	return lng
}

//export LL2MC_lat
func LL2MC_lat(lng, lat float64) float64 {
	var _ error
	lng, lat, _ = bd09mc.LL2MC(lng, lat)
	// fmt.Println(lng, lat, err)
	return lat
}

//export MC2LL_lng
func MC2LL_lng(lng, lat float64) float64 {
	var _ error
	lng, lat, _ = bd09mc.MC2LL(lng, lat)
	// fmt.Println(lng, lat, err)
	return lng
}

//export MC2LL_lat
func MC2LL_lat(lng, lat float64) float64 {
	var _ error
	lng, lat, _ = bd09mc.MC2LL(lng, lat)
	// fmt.Println(lng, lat, err)
	return lat
}

func main() {

}

// func main() {
// 	var lng, lat float64
// 	var err error

// 	lng, lat, err = bd09mc.LL2MC(108.95344, 34.265657)
// 	fmt.Println(lng, lat, err)

// 	// output: 1.212877343e+07 4.04024901e+06 <nil>

// 	lng, lat, err = bd09mc.MC2LL(12128773.43, 4040249.00)
// 	fmt.Println(lng, lat, err)
// 	// output: 108.95344 34.265657 <nil>
// }
