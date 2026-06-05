//index.js
const defaultAvatarUrl = 'https://mmbiz.qpic.cn/mmbiz/icTdbqWNOwNRna42FI242Lcia07jQodd2FJGIYQfG0LAJGFxM4FbnQP6yfMxBgJ0F3YRqJCJ1aPAK2dQagdusBZg/0'

Page({
  data: {
    userInfo: {
      avatarUrl: defaultAvatarUrl,
      nickName: '',
    },
    hasUserInfo: false,
    canIUseGetUserProfile: wx.canIUse('getUserProfile'),
  },

  // 选择头像
  onChooseAvatar(e) {
    this.setData({
      "userInfo.avatarUrl": e.detail.avatarUrl
    })
  },

  // 输入昵称
  onInputChange(e) {
    this.setData({
      "userInfo.nickName": e.detail.value
    })
  },

// 微信授权登录
async wxLogin() {
  const { nickName, avatarUrl } = this.data.userInfo

  if (!nickName) {
    wx.showToast({ title: '请输入昵称', icon: 'none' })
    return
  }

  wx.showLoading({ title: '正在登录...' })

  try {
    // 1. 调用云函数 userLogin
    // 注意：这里的 name 必须与你云开发控制台里的函数名完全一致
    const res = await wx.cloud.callFunction({
      name: 'userLogin', 
      data: {
        userInfo: {
          nickName: nickName,
          avatarUrl: avatarUrl
        }
      }
    })

    // 2. 解析云函数返回结果
    if (res.result && res.result.code === 0) {
      // 组装完整用户信息（包含云函数返回的 openid）
      const finalUserInfo = {
        nickName,
        avatarUrl,
        openid: res.result.data.openid
      }

      // 3. 同步到本地缓存和全局变量
      wx.setStorageSync('userInfo', finalUserInfo)
      getApp().globalData.userInfo = finalUserInfo

      wx.hideLoading()
      wx.showToast({ title: '登录成功', icon: 'success' })

      // 4. 延迟跳转，让用户看清“登录成功”的提示
      setTimeout(() => {
        wx.switchTab({
          url: '/pages/home/home'
        })
      }, 1000)

    } else {
      // 处理云函数逻辑抛出的错误（如 code: -1）
      throw new Error(res.result?.msg || '服务器内部错误')
    }

  } catch (err) {
    wx.hideLoading()
    console.error('云函数调用失败：', err)
    
    // 针对不同类型的错误给用户反馈
    wx.showModal({
      title: '登录失败',
      content: err.message || '网络连接异常，请稍后再试',
      showCancel: false
    })
  }
},

  // 游客模式
  guestLogin() {
    wx.showModal({
      title: '游客模式',
      content: '不登录将不会云端保存您的数据',
      success: (res) => {
        if (res.confirm) {
          wx.switchTab({
            url: '/pages/home/home'
          })
        }
      }
    })
  },

  // 兼容原版点击头像
  getUserProfile(e) {
    this.wxLogin()
  }
})