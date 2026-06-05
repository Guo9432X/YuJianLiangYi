# my_code/env_service.py
import requests
import os

def get_environment_context(lat, lng):
    """
    融合高德地图与和风天气接口
    """
    amap_key = os.environ.get("AMAP_KEY")
    qweather_key = os.environ.get("QWEATHER_KEY")
    
    context = {
        "location": "未知区域",
        "weather": {"condition": "未知", "temp": "25", "humidity": "60"}
    }
    
    try:
        # 高德逆地理编码
        geo_url = f"https://restapi.amap.com/v3/geocode/regeo?location={lng},{lat}&key={amap_key}"
        geo_res = requests.get(geo_url).json()
        if geo_res['status'] == '1':
            context["location"] = geo_res['regeocode']['addressComponent']['district']
            
        # 和风天气实时数据
        weather_url = f"https://devapi.qweather.com/v7/weather/now?location={lng},{lat}&key={qweather_key}"
        w_res = requests.get(weather_url).json()
        if w_res['code'] == '200':
            context["weather"] = {
                "condition": w_res['now']['text'],
                "temp": w_res['now']['temp'],
                "humidity": w_res['now']['humidity']
            }
    except Exception as e:
        print(f"环境信息获取异常: {e}")
        
    return context
